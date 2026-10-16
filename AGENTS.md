# AGENTS.md

代码智能体提供本仓库的协作指南。

## 项目概述

这是一个基于 Python 3.12 的 RAG + Agent 实验项目。当前技术栈：

- LlamaIndex：负责文档解析、分块、Retriever 与 QueryEngine。
- LangGraph：负责 Agent 状态、流程控制、循环和工具调用。
- Milvus（Zilliz 云北京区，官方 `MilvusClient` v2 API）：向量存储与检索。
- 火山方舟（豆包）：Chat 模型（OpenAI 兼容 SDK 接入）与 Embedding 模型（HTTP 直调，2560 维）。
- uv：负责 Python 版本、虚拟环境和依赖管理。

单元测试可离线运行（测试替身代替外部服务）；端到端问答依赖 `.env` 中的真实密钥。

## 环境与常用命令

- 项目 Python 版本由 `.python-version` 指定为 3.12。
- 使用 `uv` 管理环境和依赖；不要通过 `pip install` 修改项目依赖。
- 如果本机尚未安装所需 Python 版本，先运行 `uv python install 3.12`。

```bash
uv sync                       # 根据项目配置和锁文件创建/同步 .venv
uv run python main.py         # 在项目虚拟环境中运行 CLI（ingest/ask 子命令）
uv add <package>              # 添加运行时依赖
uv add --dev <package>        # 添加开发依赖
uv remove <package>           # 移除依赖
```

测试、Lint 与格式化：

```bash
uv run pytest                 # 单元测试（离线运行，不需要真实密钥）
uv run ruff check .           # Lint
uv run ruff format .          # 格式化
```

类型检查使用用户级 pyright（`pyright src/ tests/`）。项目通过 `.env` 文件配置密钥与参数（模板见 `.env.example`），不要假设系统环境变量已配置。

项目还定义了 `rag-agent-playground` 命令行入口，指向 `rag_agent:main`（Python 包名为 `rag_agent`，通过 `[tool.uv.build-backend] module-name` 与项目名 `rag-agent-playground` 关联）。

## 项目结构

- `main.py`：顶层 CLI 入口（`ingest` / `ask` 子命令）。
- `src/rag_agent/`：Python 项目包，按板块分两个子包，共享的配置/日志/CLI 留在包根。
  - `config.py` / `log.py` / `cli.py`：配置管理（读 `.env`）、日志与命令行。
  - `rag/`：向量存储与检索板块——`ingest.py`（文档分块）、`ark_embedding.py`（方舟 embedding）、`milvus_store.py`（Milvus 存取）、`retriever.py`（检索/QueryEngine）。
  - `agent/`：Agent 板块——`graph.py`（LangGraph 状态图）、`tools.py`（`rag_search` 工具）。
- `tests/`：离线单元测试（httpx MockTransport、FakeMilvusClient、FakeChatModel）。
- `data/sample_docs/`：端到端验证用的示例文档。
- `pyproject.toml`：项目元数据、依赖与 pytest/ruff 配置（包名 `rag_agent` 通过 `[tool.uv.build-backend] module-name` 指定）。
- `uv.lock`：锁定解析后的依赖版本。
- `.env.example`：环境变量模板。
- `.python-version`：指定项目使用的 Python 版本。

新增模块按归属板块放入 `rag/` 或 `agent/` 子包；跨板块共享的基础模块（配置、日志等）放在包根。单元测试放在 `tests/`。用户可见的安装步骤、配置或行为发生变化时，请同步更新 `README.md`。

## 协作约定

- 优先提交符合项目阶段、范围清晰的改动；不要将规划中的功能描述为已实现。
- 文档和面向用户的说明以中文为主；Python 模块、类、函数和变量使用清晰易懂的英文名称。
- 注释和文档字符串以中文为主，面向 Python 初学者编写：每个模块、类和函数/方法都应有 docstring，说明用途、关键参数与返回值；函数内部在外部调用、数据转换、条件分支等关键步骤处添加行内注释，说明这一步在做什么、为什么。
- 对首次出现的 Python 语言特性或框架用法（如 dataclass、Protocol、装饰器、LlamaIndex/LangGraph 概念），用一两句话简要解释，帮助新手理解；避免逐字复述代码字面意思的无效注释。
- 注意：LangChain `@tool` 等场景下，函数 docstring 会作为工具描述发送给模型，应保持对模型清晰、稳定的表述，不要改写成实现细节。
- 通过 `uv` 管理依赖，依赖声明写入 `pyproject.toml`，并提交对应的 `uv.lock` 更新。
- API Key 等敏感信息从环境变量读取。不要提交凭据或真实密钥。新增必需的环境变量时，在 `README.md` 中说明变量名及用途；`.env` 文件已由 Git 忽略。
- 仓库尚未声明许可证；不要在代码或文档中擅自假设项目采用某种许可证。
