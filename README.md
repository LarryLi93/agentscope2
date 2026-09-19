# AgentScope 2.0 教程

这是一套面向初学者的 AgentScope 2.0 中文实战教程，覆盖安装配置、模型接入、工具调用、技能、上下文压缩、长期记忆、RAG，以及完整的 RAG 问答助手项目。

## 目录结构

```text
.
├── articles/                 # 掘金文章 Markdown 源稿
│   └── media/                # 文章图片
├── assets/                   # 教程配套截图与图示
└── codes/                    # 示例代码与完整实战项目
    ├── ai_assistant/         # RAG 问答助手（带 Web UI）
    ├── corpus/               # RAG 示例语料
    ├── workspace_reme/       # ReMe 本地运行数据（已忽略，不提交）
    └── xhs_skill/            # 小红书爆款标题技能示例
```

## 文章与代码对应关系

| 章节 | 主题 | 相关代码 |
| --- | --- | --- |
| 01 | AgentScope 2.0 是什么 | - |
| 02 | 安装 AgentScope 2.0 | `codes/verify_install.py`、`codes/console.py` |
| 03 | 切换国内模型 | `codes/config.py`、`codes/configure_domestic_model.py` |
| 04 | 内置文件工具 | `codes/file_tools.py` |
| 05 | 创建 AnySearch 搜索工具 | `codes/anysearch_tool.py` |
| 06 | 小红书爆款标题生成技能 | `codes/xhs_skill/` |
| 07 | 上下文压缩 | `codes/context_compression.py` |
| 08 | ReMe 长期记忆 | `codes/reme_memory.py` |
| 09 | RAG 检索增强生成 | `codes/rag.py`、`codes/corpus/` |
| 10 | 总结、排错与学习路线 | - |
| 11 | 完整 RAG 问答助手 | `codes/ai_assistant/` |

## 环境准备

建议使用 Python 3.11 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

根据需要安装依赖：

```bash
# 基础示例
pip install agentscope python-dotenv

# 完整 RAG 助手
pip install -r codes/ai_assistant/requirements.txt
```

## 配置环境变量

本仓库不包含真实密钥。请在本地通过环境变量配置：

```bash
export OPENFLOWLY_BASE_URL="https://www.openflowly.com/v1"
export OPENFLOWLY_API_KEY="你的网关密钥"
export OPENFLOWLY_CHAT_MODEL="Qwen 3.7 Flash"
export OPENFLOWLY_EMBED_MODEL="Bge-m3"
export OPENFLOWLY_EMBED_DIM="1024"
export ANYSEARCH_API_KEY="你的 AnySearch 密钥"
```

如果你使用其他 OpenAI 兼容服务，只需要替换 Base URL、模型名和密钥。

## 快速开始

```bash
# 验证安装
python codes/verify_install.py

# 终端对话示例
python codes/console.py

# 文件工具示例
python codes/file_tools.py

# 上下文压缩示例
python codes/context_compression.py

# 长期记忆示例
python codes/reme_memory.py

# RAG 示例
python codes/rag.py
```

启动完整 RAG 问答助手：

```bash
cd codes/ai_assistant
pip install -r requirements.txt
uvicorn app.main:app --reload
```

然后在浏览器中打开终端提示的本地地址。

## 安全说明

- 不要把 `.env`、真实 API Key 或本地密钥提交到仓库。
- `codes/workspace_reme/`、`codes/ai_assistant/data/` 等运行时目录已在 `.gitignore` 中忽略。
- 如果曾经把真实密钥写入代码或上传到远程仓库，应立即在对应平台轮换密钥。

## License

仅用于学习和教程演示。
