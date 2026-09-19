# -*- coding: utf-8 -*-
"""RAG 问答助手的核心引擎。

职责：
- 全局装配：聊天/嵌入模型、持久化 Qdrant 向量库、KnowledgeBase、
  RAGMiddleware（知识库问答）、ReMeMiddleware（跨会话长期记忆）、AnySearch 工具；
- 多会话：每个会话一个绑定 session_id 的 Agent，共享同一份知识库与记忆工作区；
- 文档上传：按扩展名选解析器，切块后增量写入知识库；
- 流式对话：把 Agent 的 reply_stream 事件转成前端可消费的结构。

设计上 user_id 固定为 "1"（单用户演示）：
- 知识库按用户共享，所有会话都能检索到上传的文档；
- ReMe 工作区按用户共享，不同 session_id 之间可以互相召回记忆。
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import sys
import time
import uuid
import pathlib
from dataclasses import dataclass, field

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.embedding import OpenAIEmbeddingModel
from agentscope.event import TextBlockDeltaEvent, ToolCallStartEvent
from agentscope.message import UserMsg
from agentscope.middleware import RAGMiddleware, ReMeMiddleware
from agentscope.model import OpenAIChatModel
from agentscope.rag import (
    ApproxTokenChunker,
    KnowledgeBase,
    PDFParser,
    QdrantStore,
    TextParser,
    WordParser,
)
from agentscope.state import AgentState
from agentscope.tool import Toolkit

# 复用 codes/config.py
_CODES_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(_CODES_DIR) not in sys.path:
    sys.path.insert(0, str(_CODES_DIR))

from config import (  # noqa: E402
    API_KEY,
    BASE_URL,
    CHAT_MODEL,
    EMBED_DIM,
    EMBED_MODEL,
)
from app.search_tool import AnySearchTool  # noqa: E402

USER_ID = "1"
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
QDRANT_DIR = DATA_DIR / "qdrant"
REME_DIR = DATA_DIR / "reme"
COLLECTION = "user1_kb"

logging.getLogger("reme").setLevel(logging.ERROR)

# ----------------------------------------------------------------------
# ReMe 兼容补丁：当前 PyPI 的 reme 包缺少 dream_topics_step 组件，而
# ReMeMiddleware 默认的「每日摘要」流水线引用了它，会导致中间件启动失败。
# 运行时把该步骤移除，不影响 auto_memory 写回与 memory_search 检索。
# ----------------------------------------------------------------------
from agentscope.middleware._longterm_memory._reme import _config as _reme_cfg  # noqa: E402

_orig_dream_steps = _reme_cfg._dream_steps


def _patched_dream_steps() -> list[dict]:
    return [s for s in _orig_dream_steps() if s["backend"] != "dream_topics_step"]


_reme_cfg._dream_steps = _patched_dream_steps


SYSTEM_PROMPT = (
    "你是用户的 RAG 问答助手，按以下规则工作：\n"
    "1. 回答与上传资料相关的问题前，先调用 search_knowledge 检索知识库，"
    "并严格依据检索到的内容回答，可在结尾注明来源文档；\n"
    "2. 需要最新资讯、实时信息而知识库没有时，调用 AnySearch 联网搜索；\n"
    "3. 涉及用户的偏好、历史决定或过去对话时，调用 memory_search 回忆；\n"
    "4. 用简洁中文回答，列表、对比与代码用 Markdown 组织，不要编造资料里没有的内容。"
)


@dataclass
class Session:
    sid: str
    agent: Agent
    title: str = "新对话"
    created_at: float = field(default_factory=time.time)
    messages: list[dict] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class AssistantEngine:
    """全局单例引擎，FastAPI lifespan 中 startup/shutdown。"""

    def __init__(self) -> None:
        self.credential: OpenAICredential | None = None
        self.chat_model: OpenAIChatModel | None = None
        self.embedding_model: OpenAIEmbeddingModel | None = None
        self.store: QdrantStore | None = None
        self.knowledge: KnowledgeBase | None = None
        self.rag_mw: RAGMiddleware | None = None
        self.reme_mw: ReMeMiddleware | None = None
        self.tools: list = []
        self.chunker = ApproxTokenChunker(
            parameters=ApproxTokenChunker.Parameters(chunk_size=256, overlap=32),
        )
        self.sessions: dict[str, Session] = {}

    # ---------------- 启动 / 关闭 ----------------
    async def startup(self) -> None:
        for d in (UPLOAD_DIR, QDRANT_DIR, REME_DIR):
            d.mkdir(parents=True, exist_ok=True)

        self.credential = OpenAICredential(api_key=API_KEY, base_url=BASE_URL)
        self.chat_model = OpenAIChatModel(
            credential=self.credential,
            model=CHAT_MODEL,
            stream=True,
            context_size=128000,
        )
        self.embedding_model = OpenAIEmbeddingModel(
            credential=self.credential,
            model=EMBED_MODEL,
            dimensions=EMBED_DIM,
        )

        # 持久化向量库：重启后已上传文档仍在
        self.store = QdrantStore(path=str(QDRANT_DIR))
        await self.store.__aenter__()
        self.knowledge = KnowledgeBase(
            name="assistant-kb",
            description="用户通过网页上传的文档知识库。",
            embedding_model=self.embedding_model,
            vector_store=self.store,
            collection=COLLECTION,
        )
        ensured = self.knowledge.ensure_collection()
        if inspect.isawaitable(ensured):
            await ensured

        self.rag_mw = RAGMiddleware(
            knowledge_bases=[self.knowledge],
            parameters=RAGMiddleware.Parameters(mode="agentic", top_k=3),
        )
        self.reme_mw = ReMeMiddleware(
            workspace_dir=str(REME_DIR),
            parameters=ReMeMiddleware.Parameters(
                chat_model=self.chat_model,
                embedding_model=self.embedding_model,
                mode="both",
                top_k=5,
            ),
        )

        # 三类工具：知识库检索 + 长期记忆检索 + 联网搜索
        self.tools = [
            *(await self.rag_mw.list_tools()),
            *(await self.reme_mw.list_tools()),
            AnySearchTool(),
        ]

        print(
            f"[启动] 知识库集合 {COLLECTION} 已就绪；"
            f"ReMe 工作区 {REME_DIR.relative_to(BASE_DIR)}",
            flush=True,
        )

    async def shutdown(self) -> None:
        try:
            await self.reme_mw.close()
        except Exception:
            pass
        try:
            await self.store.__aexit__(None, None, None)
        except Exception:
            pass

    # ---------------- 会话管理 ----------------
    def create_session(self) -> Session:
        sid = uuid.uuid4().hex[:12]
        agent = Agent(
            name="assistant",
            system_prompt=SYSTEM_PROMPT,
            model=self.chat_model,
            toolkit=Toolkit(tools=list(self.tools)),
            middlewares=[self.rag_mw, self.reme_mw],
            state=AgentState(session_id=f"user{USER_ID}-{sid}"),
        )
        sess = Session(sid=sid, agent=agent)
        self.sessions[sid] = sess
        return sess

    def get_session(self, sid: str) -> Session | None:
        return self.sessions.get(sid)

    def list_sessions(self) -> list[dict]:
        items = [
            {
                "sid": s.sid,
                "title": s.title,
                "created_at": s.created_at,
                "message_count": len(s.messages),
            }
            for s in self.sessions.values()
        ]
        return sorted(items, key=lambda x: x["created_at"], reverse=True)

    def delete_session(self, sid: str) -> bool:
        return self.sessions.pop(sid, None) is not None

    # ---------------- 知识库 / 上传 ----------------
    @staticmethod
    def _pick_parser(filename: str):
        name = filename.lower()
        if name.endswith(".pdf"):
            return PDFParser()
        if name.endswith(".docx"):
            return WordParser()
        if name.endswith((".md", ".markdown", ".txt", ".text")):
            return TextParser()
        raise ValueError("仅支持 .md / .txt / .pdf / .docx 文件")

    async def add_document(self, filename: str, raw: bytes) -> dict:
        safe = filename.replace("/", "_").replace("\\", "_")
        path = UPLOAD_DIR / safe
        path.write_bytes(raw)

        parser = self._pick_parser(safe)
        sections = await parser.parse(file=str(path), filename=safe)
        chunks = await self.chunker.chunk(sections)
        doc_id = await self.knowledge.insert_document(
            chunks,
            document_metadata={"filename": safe},
        )
        return {"filename": safe, "doc_id": doc_id, "chunks": len(chunks)}

    async def list_documents(self) -> list[dict]:
        out = []
        for s in await self.knowledge.list_documents():
            out.append(
                {
                    "doc_id": s.document_id,
                    "filename": (s.source or {}).get("filename")
                    if isinstance(s.source, dict)
                    else getattr(s, "source", None),
                    "chunks": s.chunk_count,
                }
            )
        return out

    # ---------------- 流式对话 ----------------
    async def chat_stream(self, sess: Session, text: str):
        """产出 (event, data) 元组：tool / token / done / error。"""
        async with sess.lock:
            sess.messages.append({"role": "user", "content": text})
            if sess.title == "新对话":
                sess.title = text.strip()[:20] or "新对话"

            full = ""
            try:
                async for ev in sess.agent.reply_stream(
                    inputs=UserMsg(name="user", content=text),
                ):
                    if isinstance(ev, ToolCallStartEvent):
                        yield ("tool", {"name": ev.tool_call_name})
                    elif isinstance(ev, TextBlockDeltaEvent):
                        full += ev.delta
                        yield ("token", {"delta": ev.delta})
            except Exception as exc:  # 把异常转成一帧错误，避免前端挂起
                yield ("error", {"message": f"模型调用失败：{exc}"})
                return

            sess.messages.append({"role": "assistant", "content": full})

            # 让本轮新写入的 ReMe 记忆立即可被检索（生产环境由后台任务完成）
            try:
                # pylint: disable=protected-access
                await self.reme_mw._run_job("reindex")
            except Exception:
                pass

            yield ("done", {"title": sess.title})


# 模块级单例
engine = AssistantEngine()
