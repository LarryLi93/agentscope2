# -*- coding: utf-8 -*-
"""
第 4 节 · 内置文件工具：Read / Write / Edit

第一部分：脱离智能体直接调用工具（上手最快）；
第二部分：把工具装进 Agent，让模型自主完成「写 → 读 → 改」闭环。

要点：
- Read 是只读工具，Write / Edit 强制「先读后写」（目标文件必须先 Read 过）。
- Edit 是精确字符串替换：old_string 找不到或不唯一会失败（replace_all=True 除外）。
- 直接调用时，结果是一个 ToolChunk，文本内容在 content[0].text。

运行：
    python file_tools.py
"""
import asyncio
import os

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.permission import AdditionalWorkingDirectory, PermissionMode
from agentscope.tool import Edit, Read, Toolkit, Write

from config import API_KEY, BASE_URL, CHAT_MODEL

WORKDIR = "/tmp/as_file_demo"


# ---------------- 第一部分：直接调用 ----------------
async def part1_direct_calls() -> None:
    os.makedirs(WORKDIR, exist_ok=True)
    read = Read()
    write = Write()
    edit = Edit()

    file_path = os.path.join(WORKDIR, "notes.txt")

    # 1) 写入
    result = await write(file_path=file_path, content="Hello AgentScope\n第二行内容\n")
    print("[Write]", result.content[0].text)

    # 2) 读取（带行号）
    result = await read(file_path=file_path)
    print("[Read]")
    print(result.content[0].text)

    # 3) 精确替换
    result = await edit(
        file_path=file_path,
        old_string="第二行内容",
        new_string="被 Edit 修改后的内容",
    )
    print("[Edit]", result.content[0].text)

    # 4) 再次读取验证
    result = await read(file_path=file_path)
    print("[Read again]")
    print(result.content[0].text)


# ---------------- 第二部分：让智能体自主使用 ----------------
async def part2_agent() -> None:
    credential = OpenAICredential(api_key=API_KEY, base_url=BASE_URL)
    model = OpenAIChatModel(
        credential=credential,
        model=CHAT_MODEL,
        stream=True,
    )

    agent = Agent(
        name="file-agent",
        system_prompt="你是文件管理助手，使用 Read/Write/Edit 工具完成用户的要求。",
        model=model,
        toolkit=Toolkit(tools=[Read(), Write(), Edit()]),
    )
    # 让 Write/Edit 在 workdir 内自动放行（教学演示；生产环境按安全策略配置）
    agent.state.permission_context.mode = PermissionMode.ACCEPT_EDITS
    agent.state.permission_context.working_directories[WORKDIR] = (
        AdditionalWorkingDirectory(path=WORKDIR, source="file-tools-demo")
    )

    user_msg = UserMsg(
        name="user",
        content=(
            f"请在我的工作目录 {WORKDIR} 中完成以下任务：\n"
            "1. 创建文件 todo.txt，内容为三行：买菜、写周报、健身；\n"
            "2. 读取该文件，把内容念给我听；\n"
            "3. 把『写周报』改为『写 AgentScope 教程』；\n"
            "4. 再次读取文件，确认修改结果，并汇报你每一步做了什么。"
        ),
    )
    reply = await agent.reply(user_msg)
    text = "".join(b.text for b in reply.content if b.type == "text")
    print("\n[Agent 最终回复]\n", text)


async def main() -> None:
    print("======== 第一部分：直接调用文件工具 ========\n")
    await part1_direct_calls()
    print("\n======== 第二部分：智能体自主操作文件 ========\n")
    await part2_agent()


if __name__ == "__main__":
    asyncio.run(main())
