# -*- coding: utf-8 -*-
"""AnySearch 联网搜索工具（与第 5 节实现一致，只读、免授权、可并发）。

这里单独抽出，供 Web 助手的 Agent 挂载。接口密钥统一从 codes/config.py 读取。
"""
import sys
import pathlib

import httpx

from agentscope.message import TextBlock
from agentscope.permission import (
    PermissionBehavior,
    PermissionContext,
    PermissionDecision,
)
from agentscope.tool import ToolBase, ToolChunk

# codes/ 目录加入 import 路径，复用同一份 config.py
_CODES_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(_CODES_DIR) not in sys.path:
    sys.path.insert(0, str(_CODES_DIR))

from config import ANYSEARCH_KEY, ANYSEARCH_URL  # noqa: E402


async def anysearch_search(
    query: str,
    max_results: int = 5,
    zone: str = "cn",
    language: str = "zh-CN",
) -> str:
    """调用 AnySearch 统一搜索接口，返回可读的标题/链接/摘要文本。"""
    payload = {
        "query": query,
        "max_results": max_results,
        "zone": zone,
        "language": language,
    }
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


class AnySearchTool(ToolBase):
    """面向 Agent 的网页搜索工具：只读、允许并发、权限直接放行。"""

    name = "AnySearch"
    description = (
        "Search the web for real-time information via AnySearch. "
        "Use it when the user asks about latest news, current events, "
        "or facts not covered by the uploaded knowledge base."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
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
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="Web search is read-only.",
        )

    async def call(self, query: str, max_results: int = 5) -> ToolChunk:
        results = await anysearch_search(query, max_results=max_results)
        return ToolChunk(content=[TextBlock(text=results)])
