# -*- coding: utf-8 -*-
"""
第 3 节 · 切换国内模型：用 OpenAI 兼容接口接入 AgentScope 2.0

要点：
1. OpenAICredential 承载 api_key 与 base_url，一个凭证对应一个网关。
2. OpenAIChatModel 与 OpenAIChatFormatter 按 OpenAI Chat Completions 协议通信。
3. 国内网关普遍兼容 OpenAI 协议，因此只要改 base_url + model 即可切换。
4. 实测服务要求流式（stream=True），这也是 AgentScope 的默认值。

运行：
    python configure_domestic_model.py
"""
import asyncio

from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel

from config import API_KEY, BASE_URL, CHAT_MODEL


def build_credential() -> OpenAICredential:
    """步骤 1：创建凭证 —— 填入你的密钥与网关地址。"""
    return OpenAICredential(
        api_key=API_KEY,
        base_url=BASE_URL,  # 例如 https://www.openflowly.com/v1
    )


def build_chat_model(
    credential: OpenAICredential,
    stream: bool = True,
) -> OpenAIChatModel:
    """步骤 2：创建聊天模型。stream 默认 True，便于展示增量输出。"""
    return OpenAIChatModel(
        credential=credential,
        model=CHAT_MODEL,          # 例如 "Qwen 3.7 Flash"
        stream=stream,
        context_size=128000,       # 上下文窗口大小（上下文压缩会用到）
    )


async def demo_stream(credential: OpenAICredential) -> None:
    """流式调用：逐块观察模型的增量输出，最后一帧为完整内容。"""
    model = build_chat_model(credential, stream=True)
    msgs = [UserMsg(name="user", content="请用一句话介绍 AgentScope。")]

    print(">>> 流式响应（delta / final）：")
    async for chunk in await model(msgs):
        if chunk.is_last:
            text = "".join(
                b.text for b in chunk.content if b.type == "text"
            )
            print(f"[final] {text}")
        else:
            deltas = "".join(
                b.text for b in chunk.content if b.type == "text"
            )
            if deltas:
                print(f"[delta] {deltas}")


async def demo_non_stream(credential: OpenAICredential) -> None:
    """非流式调用：一次拿到完整响应（部分网关可能不支持，视服务而定）。"""
    model = build_chat_model(credential, stream=False)
    msgs = [UserMsg(name="user", content="1 + 1 = ?")]

    response = await model(msgs)
    text = "".join(b.text for b in response.content if b.type == "text")
    print(">>> 非流式响应：", text)


async def main() -> None:
    credential = build_credential()
    print(f"网关: {BASE_URL}\n模型: {CHAT_MODEL}\n")
    await demo_stream(credential)
    print()
    try:
        await demo_non_stream(credential)
    except Exception as exc:  # noqa: BLE001
        print(f"非流式调用失败（部分网关仅支持流式）：{exc}")


if __name__ == "__main__":
    asyncio.run(main())
