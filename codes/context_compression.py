# -*- coding: utf-8 -*-
"""
第 7 节 · 上下文压缩：让长对话始终待在模型窗口内

原理：AgentScope 通过 ContextConfig 控制两套自动机制——
1. 上下文压缩：token 用量超过 trigger_ratio × context_size 时，
   把较早消息汇总成结构化摘要，保留最近 reserve_ratio 的消息；
2. 工具结果截断：单条工具结果超过 tool_result_limit 时截断。

为了让演示快速可见，本示例把模型 context_size 故意调小（3000 token），
连续投喂较长的消息，观察上下文长度「涨 → 触发压缩 → 回落」的过程。

运行：
    python context_compression.py
"""
import asyncio

from agentscope.agent import Agent, ContextConfig
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

from config import API_KEY, BASE_URL, CHAT_MODEL

# 故意把窗口调小，几轮对话即可触发压缩（生产环境用真实窗口如 128000）
SMALL_CONTEXT = 3000


def long_message(index: int) -> str:
    """生成一段较长、带编号的话题内容，便于观察压缩后摘要保留了哪些信息。"""
    topic = (
        "正在讨论的项目材料清单：编号 {i} 包含需求分析、架构设计、接口契约、"
        "测试用例、部署方案与运维手册六个部分，其中接口契约需要团队评审，"
        "部署方案依赖测试环境完成验收。请记住这条材料的要点。"
    )
    return topic.format(i=index)


async def main() -> None:
    credential = OpenAICredential(api_key=API_KEY, base_url=BASE_URL)
    model = OpenAIChatModel(
        credential=credential,
        model=CHAT_MODEL,
        stream=True,
        context_size=SMALL_CONTEXT,  # 上下文窗口（压缩阈值以此计算）
    )

    agent = Agent(
        name="compress-demo",
        system_prompt="你是一个耐心的助手，请逐条记住用户给出的材料要点并简单回应。",
        model=model,
        toolkit=Toolkit(),  # 无工具，聚焦上下文压缩本身
        context_config=ContextConfig(
            trigger_ratio=0.8,       # 用量超过 80% × 3000 = 2400 token 时触发压缩
            reserve_ratio=0.1,       # 压缩后保留最近 10% 的消息
            tool_result_limit=800,   # 单条工具结果截断阈值（本示例无工具）
            compression_fallback_to_truncation=True,
        ),
    )

    print(f"模型 context_size = {SMALL_CONTEXT}，压缩阈值 ≈ {int(SMALL_CONTEXT * 0.8)} token\n")

    for i in range(1, 13):
        await agent.reply(UserMsg(name="user", content=long_message(i)))
        n_msgs = len(agent.state.context)
        print(f"第 {i:>2} 轮后：上下文共 {n_msgs:>3} 条消息")

    # 观察压缩后留下的摘要与最近消息
    print("\n===== 压缩后的上下文结构 =====")
    for msg in agent.state.context[-6:]:
        blocks = getattr(msg, "content", [])
        if isinstance(blocks, list) and blocks:
            first = blocks[0]
            snippet = getattr(first, "text", None) or str(first)[:80]
        else:
            snippet = str(msg)[:80]
        print(f"- {type(msg).__name__}: {snippet[:120]}")

    # 手动压缩：注入指令引导摘要重点（例如要求保留编号与材料名）
    print("\n===== 手动压缩（注入指令）=====")
    from agentscope.message import HintBlock

    await agent.compress_context(
        instructions=HintBlock(
            hint="压缩时务必保留每条材料的编号与名称，便于后续按编号检索。",
        ),
    )
    print(f"手动压缩后：上下文共 {len(agent.state.context)} 条消息")

    # 压缩后继续对话，验证摘要是否承载了旧信息
    reply = await agent.reply(
        UserMsg(name="user", content="请根据你记住的内容，说出编号 3 的材料有哪些部分。")
    )
    text = "".join(b.text for b in reply.content if b.type == "text")
    print("\n[压缩后回答] ", text)


if __name__ == "__main__":
    asyncio.run(main())
