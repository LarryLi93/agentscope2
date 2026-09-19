"""verify_install.py — 安装验证：确认 AgentScope 核心模块全部可导入。"""
from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, Read, Write, Edit
from agentscope.middleware import ReMeMiddleware, RAGMiddleware
from agentscope.rag import KnowledgeBase, QdrantStore

print("AgentScope 核心模块导入成功 ✅")
print("Agent / Credential / Model / Toolkit / 文件工具 / ReMe / RAG 全部可用")
