# -*- coding: utf-8 -*-
"""
第 6 节 · 创建「小红书爆款标题生成」技能（Skill）

Skill 是 Markdown 格式的指令集：只需一个带 frontmatter 的 SKILL.md 文件，
无需写代码即可为智能体增加新能力。AgentScope 会自动注册「Skill 查看器」，
智能体先用查看器读取指令，再按指令执行。

目录结构：
    codes/xhs_skill/
    ├── skills/xhs-title/SKILL.md   # 技能本体（frontmatter + 指令）
    └── run_skill_demo.py           # 本文件

运行：
    python run_skill_demo.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.skill import LocalSkillLoader
from agentscope.tool import Toolkit

from config import API_KEY, BASE_URL, CHAT_MODEL

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


async def main() -> None:
    credential = OpenAICredential(api_key=API_KEY, base_url=BASE_URL)
    model = OpenAIChatModel(
        credential=credential,
        model=CHAT_MODEL,
        stream=True,
    )

    agent = Agent(
        name="xhs-assistant",
        system_prompt=(
            "你是小红书内容创作助手。当用户给出笔记主题时，"
            "先读取可用的 xhs-title 技能，再严格按技能要求输出标题。"
        ),
        model=model,
        # 扫描子目录以发现 skills/ 下的 xhs-title 技能
        toolkit=Toolkit(
            skills_or_loaders=[
                LocalSkillLoader(directory=SKILLS_DIR, scan_subdir=True),
            ]
        ),
    )

    reply = await agent.reply(
        UserMsg(
            name="user",
            content="主题：用 AgentScope 2.0 做小红书 AI 内容助手。请为我生成爆款标题。",
        )
    )
    text = "".join(b.text for b in reply.content if b.type == "text")
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
