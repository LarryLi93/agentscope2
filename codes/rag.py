# -*- coding: utf-8 -*-
"""
第 9 节 · RAG 检索增强生成：三步建库 + 智能体问答

AgentScope 的 RAG 由可独立替换的模块组成：
解析器 Parser → 切块器 Chunker → 嵌入模型 → 向量库 → KnowledgeBase 句柄。

本示例流程：
1. 用 TextParser 解析三篇 Markdown 文档；
2. 用 ApproxTokenChunker 切成 Chunk；
3. 用 OpenAIEmbeddingModel（Bge-m3）嵌入，写入内存版 Qdrant；
4. 用 KnowledgeBase 检索 + 文档管理；
5. 用 RAGMiddleware 把检索接进 Agent，让模型基于资料回答。

运行：
    python rag.py
"""
import asyncio
import os

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.embedding import OpenAIEmbeddingModel
from agentscope.message import UserMsg
from agentscope.middleware import RAGMiddleware
from agentscope.model import OpenAIChatModel
from agentscope.rag import ApproxTokenChunker, KnowledgeBase, QdrantStore, TextParser
from agentscope.tool import Toolkit

from config import API_KEY, BASE_URL, CHAT_MODEL, EMBED_DIM, EMBED_MODEL

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")

# 三篇小型示例文档
DOCS = {
    "cats.md": (
        "# Cats\n\n"
        "Cats sleep 12-16 hours per day. They are crepuscular, "
        "meaning they are most active at dawn and dusk.\n\n"
        "A cat's whiskers are highly sensitive and help it navigate "
        "in the dark.\n"
    ),
    "dogs.md": (
        "# Dogs\n\n"
        "Dogs are pack animals and descend from wolves. "
        "They communicate through barking, tail wagging and posture.\n\n"
        "The average lifespan of a dog is 10-13 years depending on breed.\n"
    ),
    "agentscope.md": (
        "# AgentScope 2.0\n\n"
        "AgentScope 2.0 is a production-ready agent framework "
        "developed by Alibaba. It natively supports multi-tenancy, "
        "multi-session management and distributed deployment.\n\n"
        "Its context management integrates Mem0 and ReMe "
        "for long-term memory, and its RAG module supports parsing "
        "PDF, Word, Excel and images.\n"
    ),
}


async def build_knowledge_base() -> tuple[KnowledgeBase, QdrantStore]:
    """① 解析 → ② 切块 → ③ 嵌入入库。"""
    credential = OpenAICredential(api_key=API_KEY, base_url=BASE_URL)
    embedding_model = OpenAIEmbeddingModel(
        credential=credential,
        model=EMBED_MODEL,
        dimensions=EMBED_DIM,
    )

    # ① 解析：整篇文档作为一个 Section
    parser = TextParser()
    # ② 切块：约 256 token 一块，重叠 32 token
    chunker = ApproxTokenChunker(
        parameters=ApproxTokenChunker.Parameters(chunk_size=256, overlap=32),
    )

    os.makedirs(CORPUS_DIR, exist_ok=True)
    for filename, content in DOCS.items():
        path = os.path.join(CORPUS_DIR, filename)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

    all_chunks = []
    for filename in DOCS:
        sections = await parser.parse(
            file=os.path.join(CORPUS_DIR, filename),
            filename=filename,
        )
        all_chunks.extend(await chunker.chunk(sections))
    print(f"解析+切块完成，共 {len(all_chunks)} 个 Chunk")

    # ③ 内存版 Qdrant + KnowledgeBase 句柄
    store = QdrantStore(location=":memory:")  # 生产可用 path= 或 url=
    await store.__aenter__()
    knowledge = KnowledgeBase(
        name="demo-kb",
        description="猫、狗与 AgentScope 的微型知识库。",
        embedding_model=embedding_model,
        vector_store=store,
        collection="demo-kb",
    )

    # 把每个文档作为一个 document_id 批量写入
    for filename in DOCS:
        sections = await parser.parse(
            file=os.path.join(CORPUS_DIR, filename),
            filename=filename,
        )
        chunks = await chunker.chunk(sections)
        doc_id = await knowledge.insert_document(
            chunks,
            document_metadata={"filename": filename},
        )
        print(f"  ↳ 已入库 {filename} → {doc_id}")
    return knowledge, store


async def demo_search(knowledge: KnowledgeBase) -> None:
    """向量检索 + 文档管理演示。"""
    print("\n===== 向量检索 =====")
    results = await knowledge.search(queries=["How long do cats sleep?"], top_k=2)
    for r in results:
        print(f"  score={r.score:.4f} doc={r.document_id} | {r.chunk.content.text[:60]}...")

    print("\n===== 文档管理 =====")
    for s in await knowledge.list_documents():
        print(f"  {s.document_id} | {s.source} | chunks={s.chunk_count}")


async def demo_rag_agent(knowledge: KnowledgeBase) -> None:
    """用 RAGMiddleware 把知识库接入 Agent（agentic 模式）。"""
    credential = OpenAICredential(api_key=API_KEY, base_url=BASE_URL)
    chat_model = OpenAIChatModel(
        credential=credential,
        model=CHAT_MODEL,
        stream=True,
    )

    rag_mw = RAGMiddleware(
        knowledge_bases=[knowledge],
        parameters=RAGMiddleware.Parameters(mode="agentic", top_k=3),
    )
    agent = Agent(
        name="rag-agent",
        system_prompt=(
            "你是知识库问答助手。回答前先调用 search_knowledge 工具检索资料，"
            "然后只依据检索到的资料回答。"
        ),
        model=chat_model,
        toolkit=Toolkit(tools=await rag_mw.list_tools()),  # 暴露 search_knowledge
        middlewares=[rag_mw],
    )
    reply = await agent.reply(
        UserMsg(name="user", content="猫一天睡多久？AgentScope 2.0 支持哪些长期记忆？")
    )
    text = "".join(b.text for b in reply.content if b.type == "text")
    print("\n[RAG Agent 回答]\n", text)


async def main() -> None:
    knowledge, store = await build_knowledge_base()
    await demo_search(knowledge)
    await demo_rag_agent(knowledge)
    await store.__aexit__(None, None, None)  # 关闭向量库连接


if __name__ == "__main__":
    asyncio.run(main())
