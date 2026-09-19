# -*- coding: utf-8 -*-
"""
AgentScope 2.0 掘金教程 · 公共配置

集中管理所有密钥与接口地址，各节示例统一从这里读取。
生产环境请务必改用环境变量注入，不要把密钥写进代码仓库。
"""
import os

# ---------------- 大模型（OpenAI 兼容接口）----------------
# 国内可直接访问的 OpenAI 兼容网关，实测模型为 Qwen 3.7 Flash（流式）。
# 若你使用 APIMart / 其他网关，只需替换 base_url 与模型名即可。
BASE_URL = os.getenv("OPENFLOWLY_BASE_URL", "https://www.openflowly.com/v1")
API_KEY = os.getenv("OPENFLOWLY_API_KEY", "")  # 通过环境变量注入网关密钥
CHAT_MODEL = os.getenv("OPENFLOWLY_CHAT_MODEL", "Qwen 3.7 Flash")

# ---------------- 嵌入模型（RAG / ReMe 向量检索用）----------------
EMBED_MODEL = os.getenv("OPENFLOWLY_EMBED_MODEL", "Bge-m3")
EMBED_DIM = int(os.getenv("OPENFLOWLY_EMBED_DIM", "1024"))

# ---------------- AnySearch 搜索接口 ----------------
ANYSEARCH_URL = "https://api.anysearch.com/v1/search"
ANYSEARCH_KEY = os.getenv("ANYSEARCH_API_KEY", "")  # 通过环境变量注入 AnySearch 密钥
