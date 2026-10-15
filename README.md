# RAG Agent Playground

基于 Python 3.12 的 RAG + Agent 实验项目，使用 [LlamaIndex](https://www.llamaindex.ai/) 处理文档解析、切分和向量检索，使用 [LangGraph](https://www.langchain.com/langgraph) 编排 Agent 状态、循环和工具调用，并使用 [uv](https://docs.astral.sh/uv/) 管理 Python 版本、虚拟环境和依赖。

> **项目状态：早期搭建阶段。** 当前仓库主要是 Python/uv 项目骨架；RAG、向量检索和 LangGraph Agent 工作流仍在规划与迭代中，尚不能视为已实现功能。

## 目标能力

- **文档处理**：通过 LlamaIndex 加载和解析文档，并按配置进行分块。
- **向量检索**：将文档块转换为向量，支持语义检索，为后续回答提供上下文。
- **Agent 编排**：通过 LangGraph 管理状态、条件分支、循环和多步骤执行。
- **工具调用**：让 Agent 根据任务选择并调用工具，再根据工具结果继续推理。
- **可迭代实验**：逐步验证检索效果、Agent 行为和不同组件配置。

## 技术栈

| 组件 | 用途 |
| --- | --- |
| Python 3.12 | 项目运行时 |
| uv | Python 版本、虚拟环境和依赖管理 |
| LlamaIndex | 文档解析、分块与向量检索 |
| LangGraph | Agent 状态管理、循环与工具调用 |

## 环境要求

- macOS、Linux 或 Windows
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 已安装
- Python 3.12（可由 uv 安装）

## 快速开始

克隆仓库后，在项目根目录执行：

```bash
# 安装项目要求的 Python 版本（若本机尚未安装）
uv python install 3.12

# 根据项目配置创建/同步虚拟环境
uv sync

# 在项目虚拟环境中运行当前入口
uv run python main.py
```

项目通过 `.python-version` 指定 Python 3.12。`uv sync` 会在项目目录下创建 `.venv` 并按 `pyproject.toml` 与 `uv.lock` 同步依赖。通常无需手动激活虚拟环境；如需激活：

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

## 依赖管理

使用 `uv` 管理依赖，不建议在项目环境中直接用 `pip install` 修改依赖：

```bash
# 添加运行时依赖
uv add <package>

# 添加开发依赖
uv add --dev <package>

# 移除依赖
uv remove <package>

# 按锁文件同步环境
uv sync
```

LlamaIndex 和 LangGraph 将在功能实现时作为项目依赖加入，并通过 `uv.lock` 锁定版本。

## 项目结构

```text
.
├── main.py                    # 当前最小运行入口
├── pyproject.toml             # 项目元数据、Python 要求和依赖声明
├── uv.lock                    # 依赖锁文件
├── .python-version             # 项目 Python 版本（3.12）
└── src/
    └── rag_agent_playground/   # 项目 Python 包
```

随着功能完善，文档加载、索引/检索、Agent 图和工具模块会逐步放入 `src/rag_agent_playground/`。

## 配置与密钥

需要模型或向量服务时，通过环境变量配置 API Key 等敏感信息。不要将密钥提交到 Git；本地 `.env` 文件已被忽略。新增环境变量时，请同步在文档中说明变量名和用途，不要写入真实密钥。

## 开发计划

- [ ] 建立文档加载、解析与分块流程
- [ ] 接入向量存储和检索
- [ ] 构建 LangGraph Agent 状态图与基本循环
- [ ] 增加工具定义、调用和执行结果回传
- [ ] 补充配置管理、日志和测试
- [ ] 使用示例文档验证端到端问答流程

## 许可证

当前仓库尚未声明许可证；在添加许可证文件前，请勿假定项目可按特定开源许可证使用或再分发。
