# -*- coding: utf-8 -*-
"""
第 5 节 · 创建 AnySearch 搜索工具

AnySearch（https://www.anysearch.com/）是专为 AI Agent 设计的搜索基础设施，
统一 POST https://api.anysearch.com/v1/search 即可完成网页/垂直领域搜索。

本节演示两种创建工具的方式：
1. FunctionTool：把普通 Python 函数一键包装成工具（最快）；
2. ToolBase 子类：完全控制工具描述、schema 与权限行为（最灵活）。

最后把工具装进 Agent，让模型自主搜索并总结。

运行：
    python anysearch_tool.py
"""
import asyncio
import json

import httpx

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.message import TextBlock, UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.permission import (
    PermissionBehavior,
    PermissionContext,
    PermissionDecision,
)
from agentscope.tool import FunctionTool, ToolBase, ToolChunk, Toolkit

from config import ANYSEARCH_KEY, ANYSEARCH_URL, API_KEY, BASE_URL, CHAT_MODEL


# ================= 1) 核心搜索函数（两个包装方式共用） =================
async def anysearch_search(
    query: str,
    max_results: int = 5,
    tag: str | None = None,
    zone: str = "cn",
    language: str = "zh-CN",
) -> str:
    """调用 AnySearch 统一搜索接口并返回可读结果。

    Args:
        query: 搜索关键词。
        max_results: 返回结果数量，1-10。
        tag: 垂直领域标签，例如 "code.doc"、"finance.quote"。
        zone: 地区，cn 或 intl。
        language: 偏好语言，例如 zh-CN。
    """
    payload: dict = {
        "query": query,
        "max_results": max_results,
        "zone": zone,
        "language": language,
    }
    if tag:
        payload["tag"] = tag

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            ANYSEARCH_URL,
            headers={
                "Authorization": f"Bearer {ANYSEARCH_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    if data.get("code") != 0:
        return f"搜索失败：{data.get('message')}"

    lines = []
    for i, item in enumerate(data["data"]["results"], 1):
        lines.append(
            f"{i}. {item.get('title', '')}\n"
            f"   URL: {item.get('url', '')}\n"
            f"   摘要: {item.get('snippet', item.get('content', ''))[:300]}"
        )
    return "\n".join(lines)


# ================= 2) 方式一：FunctionTool 包装 =================
async def demo_function_tool() -> None:
    tool = FunctionTool(anysearch_search)  # 名称/描述/schema 自动推导
    result = await tool(query="AgentScope 2.0 框架", max_results=2)
    print(">>> FunctionTool 直接调用结果：")
    print(result.content[0].text[:600])


# ================= 3) 方式二：ToolBase 子类（自定义权限） =================
class AnySearchTool(ToolBase):
    """面向 Agent 的 AnySearch 搜索工具：只读、允许并发、权限直接放行。"""

    name = "AnySearch"
    description = (
        "Search the web for real-time information via AnySearch. "
        "Use it when you need current news, docs, or facts."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results, 1-10.",
                "default": 5,
            },
        },
        "required": ["query"],
    }
    is_concurrency_safe = True
    is_read_only = True

    async def check_permissions(
        self,
        tool_input: dict,
        context: PermissionContext,
    ) -> PermissionDecision:
        # 搜索是只读操作，直接放行，避免打断智能体的推理节奏
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="Web search is read-only.",
        )

    async def call(self, query: str, max_results: int = 5) -> ToolChunk:
        results = await anysearch_search(query, max_results=max_results)
        return ToolChunk(content=[TextBlock(text=results)])


async def demo_agent() -> None:
    credential = OpenAICredential(api_key=API_KEY, base_url=BASE_URL)
    model = OpenAIChatModel(
        credential=credential,
        model=CHAT_MODEL,
        stream=True,
    )
    agent = Agent(
        name="search-agent",
        system_prompt=(
            "你是一个联网信息助手。需要最新信息时，调用 AnySearch 工具搜索，"
            "然后基于搜索结果回答用户。"
        ),
        model=model,
        toolkit=Toolkit(tools=[AnySearchTool()]),
    )

    reply = await agent.reply(
        UserMsg(
            name="user",
            content="帮我搜索一下 AgentScope 2.0 的发布信息，并总结它相比 1.0 的主要变化。",
        )
    )
    text = "".join(b.text for b in reply.content if b.type == "text")
    print(">>> Agent 联网回答：\n", text)


async def main() -> None:
    await demo_function_tool()
    print()
    await demo_agent()


if __name__ == "__main__":
    asyncio.run(main())
