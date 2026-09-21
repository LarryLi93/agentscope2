# 《AgentScope 2.0实战指南》03 - 切换国内模型（配置 URL 和密钥）

### 3.1 本节目标

本节以火山方舟（Volcengine Ark）为例，演示如何在 AgentScope 2.0 中配置模型服务、跑通第一个对话调用。把它换成任意 OpenAI 兼容的模型服务，步骤完全一致。

### 3.2 核心概念：Credential + Model 如何配置 URL 和密钥

AgentScope 把「模型接入」拆成两层：

- **Credential（凭证）**：保存 `api_key` 与 `base_url`，负责「怎么连接和认证」；
- **Model（模型）**：按 Chat Completions 协议发请求、解析响应，负责「怎么调用」。

两者解耦后，切换模型服务只改配置、不动业务代码。`base_url` 指向模型服务的 OpenAI 兼容接入点，模型的实际提供方不写在业务代码里——这也是框架把模型接入独立成层的原因。

配置一项模型服务，只需要两样信息：

- `base_url`：模型服务的 OpenAI 兼容接入地址；
- `api_key`：该服务的 API 密钥。

以后换模型，只改凭证的 `base_url` / `api_key` 与模型的 `model` 名称；中间件、工具、记忆等上层逻辑都不用动。

官方文档对模型接入的示意（红色为必填、灰色为可选）：

![LLM 模型配置](../assets/as_model_llm.png)

### 3.3 完整可运行源码

下面这段代码（源码包 `configure_model.py`）演示了「配置火山方舟 + 流式/非流式两种调用」：

```python
# configure_model.py

"""第 3 节 · 配置模型：以火山方舟（Volcengine Ark）为例接入 AgentScope 2.0"""
import asyncio

from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel

# 密钥与接入点统一放在 config.py 中管理（见源码包），这里从环境变量读取
from config import API_KEY, BASE_URL, CHAT_MODEL

def build_credential() -> OpenAICredential:
    """步骤 1：创建凭证 —— 填入 API Key 与接入地址。"""
    return OpenAICredential(
        api_key=API_KEY,              # 例如从环境变量 ARK_API_KEY 读取
        base_url=BASE_URL,            # 火山方舟 OpenAI 兼容接入点
    )

def build_chat_model(credential, stream: bool = True) -> OpenAIChatModel:
    """步骤 2：创建聊天模型。stream 默认 True，便于展示增量输出。"""
    return OpenAIChatModel(
        credential=credential,
        model=CHAT_MODEL,             # 推理接入点 ID（ep- 开头）或套餐模型 ID
        stream=stream,
        context_size=128000,          # 上下文窗口大小（第 7 节压缩会用到）
    )

async def demo_stream(credential) -> None:
    """流式调用：逐块观察模型的增量输出，最后一帧为完整内容。"""
    model = build_chat_model(credential, stream=True)
    msgs = [UserMsg(name="user", content="请用一句话介绍 AgentScope。")]

    print(">>> 流式响应（delta / final）：")
    async for chunk in await model(msgs):
        if chunk.is_last:
            text = "".join(b.text for b in chunk.content if b.type == "text")
            print(f"[final] {text}")
        else:
            deltas = "".join(b.text for b in chunk.content if b.type == "text")
            if deltas:
                print(f"[delta] {deltas}")

async def demo_non_stream(credential) -> None:
    """非流式调用：一次拿到完整响应（部分服务可能不支持，视服务而定）。"""
    model = build_chat_model(credential, stream=False)
    msgs = [UserMsg(name="user", content="1 + 1 = ?")]
    response = await model(msgs)
    text = "".join(b.text for b in response.content if b.type == "text")
    print(">>> 非流式响应：", text)

async def main() -> None:
    credential = build_credential()
    print(f"base_url: {BASE_URL}\nmodel: {CHAT_MODEL}\n")
    await demo_stream(credential)
    print()
    try:
        await demo_non_stream(credential)
    except Exception as exc:
        print(f"非流式调用失败：{exc}")

if __name__ == "__main__":
    asyncio.run(main())
```

**运行结果（终端实跑输出）：**

![configure_model.py 运行结果：流式 delta/final 与非流式响应](../assets/run_configure_domestic_model.png)

配套的 `config.py`（密钥统一管理，发布前请把密钥改为环境变量）：

```python
# config.py

import os

# 火山方舟 OpenAI 兼容接入点（华北北京）；若开通 Coding Plan 套餐则用 /api/coding/v3
BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
API_KEY = os.getenv("ARK_API_KEY", "你的方舟APIKey")   # 请替换

# model 填「推理接入点 ID」（ep- 开头）；若用 Coding Plan 等套餐则填对应模型 ID
CHAT_MODEL = os.getenv("ARK_CHAT_MODEL", "ep-2026xxxxxx-xxxxx")
EMBED_MODEL = os.getenv("ARK_EMBED_MODEL", "doubao-embedding")
EMBED_DIM = int(os.getenv("ARK_EMBED_DIM", "2560"))   # 以所选嵌入模型的实际维度为准
```

> 火山方舟的接入地址为 `https://ark.cn-beijing.volces.com/api/v3`（华北北京区域）；`api_key` 在火山方舟控制台的「API Key 管理」创建。`model` 填你在控制台创建的「推理接入点 ID」（形如 `ep-` 开头），或直接填套餐对应的模型 ID（如 `doubao-seed-2-1-pro-260628`）——两者都走同一个 `base_url`。

### 3.4 流式和非流式输出案例

上节代码同时演示了两种调用方式。它们的差异：

- **流式（stream=True）**：逐 token 返回，首字快、体验好，还能做「边生成边渲染」；AgentScope 的默认值。代码需要处理异步生成器，取 `is_last` 的帧作为完整内容。
- **非流式（stream=False）**：一次性返回完整结果，代码更简单，适合「结果整体可用」的场景（如批量离线处理）。

实测输出（以火山方舟 + `doubao-seed-2-1-pro-260628` 为例）：

```text
base_url: https://ark.cn-beijing.volces.com/api/v3
model: doubao-seed-2-1-pro-260628

> 流式响应（delta / final）：
[delta] AgentScope
[delta] 是
[delta] 由阿里云
[delta] 通义实验室开源
[delta] 的多智能体开发
[delta] 框架……
[final] AgentScope是由阿里云通义实验室开源的多智能体开发框架，旨在提供完整工具链以帮助用户高效构建、协同管理与部署基于大语言模型的AI智能体应用。

> 非流式响应： 1 + 1 = 2
```

生产建议：**面向用户的一律用流式，批处理可以非流式**。部分模型服务对非流式接口的兼容性不如流式，本文示例统一用流式，少踩一个坑。

### 3.5 模型层还有哪些参数

`OpenAIChatModel` 除了 `model` / `stream` / `context_size`，还有几个常用的生产参数：

- **`max_tokens`**：限制单次输出长度，防止模型生成过长、多烧 token；
- **`temperature`**：控制随机性，检索问答用低值（0.2 左右），创意生成用高值；
- **`formatter`**：自定义消息与响应的序列化格式，对接特殊协议时用；
- **`retry` 相关**：框架层内置重试与降级策略，网络抖动时自动重试。

这些参数在后面的示例里没有全用到，但它们是「从能跑通到能上线」的必经之路。先把第 3 节的模型调用跑通，后面每一节都建立在这个地基上。

### 3.6 如何更换协议

AgentScope 把不同厂商的接入抽象成对称的 Credential / Model 类。除了本节用的 OpenAI 兼容协议，框架还内置了其他厂商的封装，切换时同样只改配置：

| 厂商 | 凭证类 | 模型类 |
|---|---|---|
| Anthropic | `AnthropicCredential` | `AnthropicChatModel` |
| Gemini | `GeminiCredential` | `GeminiChatModel` |
| 通义系 | `DashScopeCredential` | `DashScopeChatModel` |

用法和 `OpenAICredential` 完全对称——**换协议不换上层代码**。

两点说明：

- OpenAI 兼容协议覆盖了绝大多数模型服务，一套代码即可对接不同提供方；
- OpenAI 兼容接口几乎都使用 `Bearer Token` 认证，个别平台的差异（如额外加组织 ID）通过 `OpenAICredential` 的扩展字段解决即可。

### 3.7 关键点与避坑

1. **`model` 要写对**：火山方舟填「推理接入点 ID」（`ep-` 开头）或套餐模型 ID（如 `doubao-seed-2-1-pro-260628`），以控制台为准，别照抄示例里的占位值——不同服务、不同模型的命名不同。
2. **`stream` 参数**：AgentScope 默认 `stream=True`，返回异步生成器，需要 `async for` 迭代并取 `is_last` 帧；部分服务只支持流式（非流式会报错），统一用流式最稳妥。代码里的 `try/except` 就是为这类服务准备的降级路径。
3. **`context_size` 一定要设**：它决定第 7 节上下文压缩的触发阈值，不设的话压缩机制无法正确工作。设置成模型真实窗口即可（如 128000）。
4. **密钥管理**：千万别把真实密钥写进公开仓库或文章，一律走环境变量；本文源码包内为便于本地复现保留了占位，发布前请替换。
5. **嵌入模型**：RAG / ReMe 还需要一个嵌入模型，火山方舟提供 OpenAI 兼容的 `/v1/embeddings`（本文用 `doubao-embedding`，维度以所选模型为准，见下图文档中的 Embedding 配置说明）。嵌入模型与对话模型可以在同一个凭证下共存，因为它们都走同一个 `base_url`。

![Embedding 模型配置](../assets/as_model_embedding.png)

到这里，「模型能通」这个地基就打好了。第 4 节开始，我们给这个只会聊天的模型装上工具。

源码地址：[https://github.com/LarryLi93/agentscope2](https://github.com/LarryLi93/agentscope2)
