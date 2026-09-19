# RAG 问答助手（AgentScope 2.0 + FastAPI + 原生前端）

一个带 Web 界面的 RAG 问答助手：上传文档建知识库、多会话问答、SSE 流式输出、
AnySearch 联网搜索、ReMe 跨会话长期记忆。

## 功能

- FastAPI 封装后端接口，SSE 流式返回
- 原生 HTML/CSS/JS 前端，无需构建：
  - 左侧：新建对话 + 会话列表 + 知识库面板
  - 右侧：聊天区 + 底部输入框
- 多会话隔离，每个会话独立 Agent，共享同一份知识库与长期记忆
- 上传 Markdown / TXT / PDF / Word / PPTX，自动解析、切分、向量化入库
- RAG 问答（Qdrant 本地持久化，重启不丢）
- ReMe 长期记忆：自动记住用户信息与偏好，跨会话召回
- AnySearch 联网搜索，模型按需调用
- 用户 ID 写死为 `1`（单用户演示，接入登录后替换即可）

## 目录结构

```
ai_assistant/
├── app/
│   ├── main.py          # FastAPI 应用、路由、SSE
│   ├── engine.py        # Agent 引擎：知识库 / RAG / ReMe / 会话管理
│   └── search_tool.py   # AnySearch 联网搜索工具
├── static/
│   ├── index.html       # 页面结构
│   ├── style.css        # 样式
│   └── app.js           # 会话、SSE、上传、渲染逻辑
├── data/                # 运行时数据（uploads / qdrant / reme，自动生成）
└── requirements.txt
```

模型、API Key 复用上级目录的 `codes/config.py`。

## 启动

```bash
# 在仓库根目录创建虚拟环境并安装依赖（若尚未安装）
python3 -m venv .venv
source .venv/bin/activate
pip install -r codes/ai_assistant/requirements.txt

# 进入本目录启动
cd codes/ai_assistant
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000>。

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| POST | `/api/sessions` | 新建会话 |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/{sid}` | 会话元信息 |
| DELETE | `/api/sessions/{sid}` | 删除会话 |
| GET | `/api/sessions/{sid}/messages` | 会话历史消息 |
| POST | `/api/sessions/{sid}/chat` | 提问（SSE 流式） |
| POST | `/api/documents/upload` | 上传文档（multipart，字段 `files`） |
| GET | `/api/documents` | 知识库文档列表 |

聊天接口 SSE 事件：`tool`（调用工具）、`token`（流式文本增量）、
`done`（结束）、`error`（出错）。
