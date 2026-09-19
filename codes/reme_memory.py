# -*- coding: utf-8 -*-
"""
第 8 节 · 长期记忆（ReMe 实现）：跨会话记住用户的偏好

ReMe 是 AgentScope 团队维护的「文件型记忆工具箱」，ReMeMiddleware 把它
嵌入当前进程（无需单独起服务）：
- 写回：每次回复结束后，自动调用 ReMe 的 auto_memory 任务，从对话中
  提取记忆卡片并写入 workspace_dir；
- 检索：mode="both" 时既自动检索注入，也暴露 memory_search 工具，
  由智能体按需查询；
- 跨会话：只要复用同一个 workspace_dir，新会话就能召回旧记忆。

演示流程：会话 1 让助手记住三条信息 → 手动 reindex 让新记忆立即可检索
→ 会话 2（全新上下文）询问记忆中的信息。

运行：
    python reme_memory.py
"""
import asyncio
import logging
import os
import shutil

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.embedding import OpenAIEmbeddingModel
from agentscope.event import (
    TextBlockDeltaEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
)
from agentscope.message import UserMsg
from agentscope.middleware import ReMeMiddleware
from agentscope.model import OpenAIChatModel
from agentscope.state import AgentState
from agentscope.tool import Toolkit

from config import API_KEY, BASE_URL, CHAT_MODEL, EMBED_DIM, EMBED_MODEL

WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace_reme")

# 让 ReMe 的启动日志安静一些
logging.getLogger("reme").setLevel(logging.ERROR)


# ------------------------------------------------------------------
# 兼容性补丁（说明见文末）：
# 当前 PyPI 的 reme 包尚未提供 dream_topics_step 这一步骤组件，
# 而 AgentScope 的 ReMeMiddleware 默认装配的「每日摘要」流水线里引用了
# 它，导致中间件启动失败。这里在运行时把该步骤从流水线中移除——
# 不影响本演示用到的 auto_memory 写回与 memory_search 检索。
# ------------------------------------------------------------------
from agentscope.middleware._longterm_memory._reme import _config as _reme_cfg  # noqa: E402

_ORIGINAL_DREAM_STEPS = _reme_cfg._dream_steps


def _patched_dream_steps() -> list[dict]:
    steps = _ORIGINAL_DREAM_STEPS()
    return [s for s in steps if s["backend"] != "dream_topics_step"]


_reme_cfg._dream_steps = _patched_dream_steps


def _build_models():
    credential = OpenAICredential(api_key=API_KEY, base_url=BASE_URL)
    chat_model = OpenAIChatModel(
        credential=credential,
        model=CHAT_MODEL,
        stream=True,
    )
    embedding_model = OpenAIEmbeddingModel(
        credential=credential,
        model=EMBED_MODEL,
        dimensions=EMBED_DIM,
    )
    return chat_model, embedding_model


def _build_agent(
    chat_model: OpenAIChatModel,
    mw: ReMeMiddleware,
    session_id: str,
) -> Agent:
    """构造一个绑定 session_id 的智能体。

    ReMe 按 session_id 隔离写回，检索则覆盖整个 workspace，
    因此新会话（不同 session_id）可以召回旧会话的记忆。
    """
    return Agent(
        name="assistant",
        system_prompt=(
            "你是一个贴心的助手。当问题可能依赖过去的持久信息"
            "（用户偏好、姓名、历史决定）时，请调用 memory_search 查询；"
            "记忆保存是自动的，你不需要手动写入。"
        ),
        model=chat_model,
        # both / agent_control 模式必须显式把 memory_search 放入 toolkit
        toolkit=Toolkit(tools=[]),  # 下方会覆盖为 memory.list_tools()
        middlewares=[mw],
        state=AgentState(session_id=session_id),
    )


async def _run_turn(agent: Agent, content: str) -> None:
    """流式跑一轮对话，打印智能体的工具调用与最终回复。"""
    text_parts: list[str] = []
    async for ev in agent.reply_stream(inputs=UserMsg(name="user", content=content)):
        if isinstance(ev, ToolCallStartEvent):
            print(f"  [工具调用] {ev.tool_call_name} ...")
        elif isinstance(ev, TextBlockDeltaEvent):
            text_parts.append(ev.delta)
    print("  [回答]", "".join(text_parts))


async def main() -> None:
    # 每次运行从干净工作区开始（该目录由本示例创建，可安全重置）
    shutil.rmtree(WORKSPACE_DIR, ignore_errors=True)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    print(f"ReMe 工作区：{WORKSPACE_DIR}\n")

    chat_model, embedding_model = _build_models()
    mw = ReMeMiddleware(
        workspace_dir=WORKSPACE_DIR,
        parameters=ReMeMiddleware.Parameters(
            chat_model=chat_model,       # auto_memory 抽取用
            embedding_model=embedding_model,  # 开启向量检索
            mode="both",                 # 自动检索 + memory_search 工具
            top_k=5,
        ),
    )

    # ---------- 会话 1：写入记忆 ----------
    print("===== 会话 1：告诉助手要记住的信息 =====")
    agent1 = _build_agent(chat_model, mw, session_id="alice-main")
    agent1.toolkit = Toolkit(tools=await mw.list_tools())
    await _run_turn(
        agent1,
        "请记住三件事：1）我住在杭州；2）我偏好简洁的中文回答；"
        "3）我最近在学 AgentScope 2.0。",
    )

    # auto_memory 写回后，记忆卡片还需要经过索引任务才可检索。
    # 生产环境由 ReMe 后台 watch 循环自动完成；这里手动触发一次
    # reindex，保证下一步立刻能搜到（与官方示例做法一致）。
    # pylint: disable-next=protected-access
    await mw._run_job("reindex")

    print("\n[ReMe 已持久化的记忆（按「杭州」检索）]")
    # pylint: disable-next=protected-access
    for memory in await mw._search("杭州", limit=5):
        print(f"  · {memory}")

    # ---------- 会话 2：全新上下文，召回记忆 ----------
    print("\n===== 会话 2：新会话（空上下文）询问旧记忆 =====")
    agent2 = _build_agent(chat_model, mw, session_id="bob-new")
    agent2.toolkit = Toolkit(tools=await mw.list_tools())
    await _run_turn(
        agent2,
        "你还记得我住在哪个城市、有什么回答偏好吗？最近在学什么？",
    )

    await mw.close()
    print("\n完成。ReMe 的记忆卡片保存在：", WORKSPACE_DIR)


if __name__ == "__main__":
    asyncio.run(main())
