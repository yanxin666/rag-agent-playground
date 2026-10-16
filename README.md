# RAG Agent Playground

基于 Python 3.12 的 RAG + Agent 实验项目，使用 [LlamaIndex](https://www.llamaindex.ai/) 处理文档加载、解析、分块与 QueryEngine 检索，使用 [LangGraph](https://www.langchain.com/langgraph) 编排 Agent 状态、循环与工具调用，向量库使用 [Milvus](https://milvus.io/)（Zilliz 云北京区），LLM 与 Embedding 均接入火山方舟（豆包），并使用 [uv](https://docs.astral.sh/uv/) 管理 Python 版本、虚拟环境和依赖。

## 架构

```
用户提问
  │
  ▼
LangGraph Agent（agent 节点 ⇄ ToolNode 条件循环）
  │ 调用 rag_search 工具
  ▼
LlamaIndex QueryEngine = MilvusRetriever（Ark embedding → Milvus 向量检索）+ LLM 综合
  │ 返回答案与相关原文片段（ToolMessage 回传）
  ▼
Agent 基于检索结果生成最终回答
```

- 文档入库（ingest）：`SimpleDirectoryReader` 加载 → `SentenceSplitter` 分块 → 方舟 embedding（HTTP 直调，2560 维）→ 写入 Milvus（官方 `MilvusClient` v2 API）。
- 检索（ask）：LangGraph Agent 判断是否需要检索，通过 `rag_search` 工具调用 LlamaIndex `RetrieverQueryEngine`，得到回答与原文片段后回传给 Agent。

## 技术栈

| 组件 | 用途 |
| --- | --- |
| Python 3.12 | 项目运行时 |
| uv | Python 版本、虚拟环境和依赖管理 |
| LlamaIndex | 文档解析、分块、Retriever 与 QueryEngine |
| LangGraph | Agent 状态管理、循环与工具调用 |
| pymilvus（MilvusClient） | 向量存储与检索（Zilliz 云北京区） |
| 火山方舟（豆包） | Chat 模型（OpenAI 兼容 SDK 接入）与 Embedding 模型（HTTP 直调） |
| pytest / ruff / pyright | 测试、Lint/格式化、类型检查 |

## 环境要求

- macOS、Linux 或 Windows
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 已安装
- Python 3.12（可由 uv 安装）
- 火山方舟 API Key；Zilliz 云（北京区）实例的 endpoint 与 API Key

## 快速开始

```bash
# 1. 安装依赖（会按 .python-version 安装/复用 Python 3.12）
uv sync

# 2. 准备配置文件
cp .env.example .env
# 编辑 .env，填入 ARK_API_KEY、MILVUS_URI、MILVUS_TOKEN

# 3. 将示例文档写入知识库（首次执行，重建集合）
uv run python main.py ingest data/sample_docs --rebuild

# 4. 提问验证端到端流程
uv run python main.py ask "知识库什么时候更新？"
uv run python main.py ask "RAG 的核心流程是什么？"
```

也可使用等价的控制台命令：`uv run rag-agent-playground ingest ...` / `uv run rag-agent-playground ask ...`。

## 配置说明

所有配置通过项目根目录的 `.env` 文件读取（复制 `.env.example` 后填写）；已存在的进程环境变量优先于 `.env`。不要提交真实密钥，`.env` 已被 Git 忽略。

| 环境变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `ARK_API_KEY` | 是 | - | 火山方舟 API Key（Chat 与 Embedding 共用） |
| `ARK_BASE_URL` | 否 | `https://ark.cn-beijing.volces.com/api/v3` | 方舟 OpenAI 兼容端点 |
| `ARK_CHAT_MODEL` | 否 | `doubao-seed-2-1-pro-260628` | Chat 模型名（支持 ep- 接入点 ID） |
| `ARK_EMBEDDING_MODEL` | 否 | `doubao-embedding-text-240715` | Embedding 模型名 |
| `EMBEDDING_DIM` | 否 | `2560` | 向量维度，需与模型输出一致 |
| `MILVUS_URI` | 是 | - | Zilliz 云（北京区）endpoint |
| `MILVUS_TOKEN` | 是 | - | Zilliz 云 API Key |
| `MILVUS_COLLECTION` | 否 | `rag_agent_playground` | 集合名 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 否 | `512` / `50` | 分块参数 |
| `RETRIEVAL_TOP_K` | 否 | `4` | 检索返回的文本块数 |
| `LOG_LEVEL` | 否 | `INFO` | 日志级别 |

注意：集合维度创建后不可修改；更换 Embedding 模型或维度后需 `ingest --rebuild` 重建集合并重新写入。

## 依赖管理

使用 `uv` 管理依赖，不建议在项目环境中直接用 `pip install` 修改依赖：

```bash
uv add <package>              # 添加运行时依赖
uv add --dev <package>        # 添加开发依赖
uv remove <package>           # 移除依赖
uv sync                       # 按锁文件同步环境
```

## 开发与测试

```bash
uv run pytest                 # 运行全部单元测试（离线，不需要真实密钥）
uv run ruff check .           # Lint
uv run ruff format .          # 格式化
pyright src/ tests/           # 类型检查（需自行安装：uv tool install pyright）
```

单元测试通过 httpx MockTransport、内存版 FakeMilvusClient、FakeChatModel 等测试替身离线运行；端到端问答需在 `.env` 配置真实密钥后执行上面的 ingest/ask 命令验证。

## 项目结构

```text
.
├── main.py                          # 顶层运行入口（ingest / ask）
├── pyproject.toml                   # 项目元数据、依赖与工具配置
├── uv.lock                          # 依赖锁文件
├── .env.example                     # 配置模板（复制为 .env 使用）
├── data/sample_docs/                # 示例文档（用于端到端验证）
├── src/rag_agent/                   # 项目 Python 包（共享配置/日志/CLI 在包根）
│   ├── config.py                    # 集中配置管理（.env / 环境变量）
│   ├── log.py                       # 日志初始化
│   ├── cli.py                       # 命令行入口（ingest / ask）
│   ├── rag/                         # 向量存储与检索板块
│   │   ├── ingest.py                # 文档加载、解析与分块
│   │   ├── ark_embedding.py         # 方舟 embedding（HTTP 直调）
│   │   ├── milvus_store.py          # MilvusClient 薄封装（建集合/写入/检索）
│   │   └── retriever.py             # MilvusRetriever + QueryEngine 组装
│   └── agent/                       # Agent 板块
│       ├── graph.py                 # LangGraph 状态图与条件循环
│       └── tools.py                 # rag_search 工具
└── tests/                           # 单元测试（离线）
```

## 开发计划

- [x] 建立文档加载、解析与分块流程
- [x] 接入向量存储和检索（Milvus / Zilliz 云）
- [x] 构建 LangGraph Agent 状态图与基本循环
- [x] 增加工具定义、调用和执行结果回传
- [x] 补充配置管理、日志和测试
- [x] 使用示例文档验证端到端问答流程

## 许可证

当前仓库尚未声明许可证；在添加许可证文件前，请勿假定项目可按特定开源许可证使用或再分发。
