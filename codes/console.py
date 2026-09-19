# -*- coding: utf-8 -*-
"""
终端 UI（Console）：在终端里直接和智能体对话、查看事件流。

说明：AgentScope 没有 `agentscope console` 这个 shell 命令（包本身不注册
任何命令行入口）。终端对话能力由 agentscope.console 模块提供，需要在
代码里调用 launch_console，再用 python 运行本脚本进入交互。

运行：
    python console.py
退出：
    输入 exit / quit，或按 Ctrl+D；工具确认提示处按 Ctrl+C 可终止本次回复。
工具授权：
    工具执行前会逐个询问：y = 允许一次；a = 允许并记住该规则；其他键 = 拒绝。
"""
import asyncio

from agentscope.agent import Agent
from agentscope.console import launch_console
from agentscope.credential import OpenAICredential
from agentscope.model import OpenAIChatModel
from agentscope.tool import Bash, Edit, Read, Toolkit, Write

from config import API_KEY, BASE_URL, CHAT_MODEL


async def main() -> None:
    credential = OpenAICredential(api_key=API_KEY, base_url=BASE_URL)
    model = OpenAIChatModel(
        credential=credential,
        model=CHAT_MODEL,
        stream=True,
    )

    agent = Agent(
        name="Friday",
        system_prompt="你是乐于助人的智能体助手 Friday，尽量用中文回答。",
        model=model,
        toolkit=Toolkit(tools=[Bash(), Read(), Write(), Edit()]),
    )

    # 接管终端输入循环、流式渲染与工具确认；对话状态保存在 agent.state，
    # 进程退出即结束，不做持久化。
    await launch_console(agent)


if __name__ == "__main__":
    asyncio.run(main())
