# AGENTS.md

代码智能体提供本仓库的协作指南。

## 项目概述

这是一个基于 Python 3.12 的 RAG + Agent 实验项目，目前处于早期搭建阶段。规划中的技术栈：

- LlamaIndex：负责文档解析、分块和向量检索。
- LangGraph：负责 Agent 状态、流程控制、循环和工具调用。
- uv：负责 Python 版本、虚拟环境和依赖管理。

当前仓库仍是项目骨架，RAG 和 Agent 核心功能尚未实现，LlamaIndex 和 LangGraph 也尚未加入项目依赖。

## 环境与常用命令

- 项目 Python 版本由 `.python-version` 指定为 3.12。
- 使用 `uv` 管理环境和依赖；不要通过 `pip install` 修改项目依赖。
- 如果本机尚未安装所需 Python 版本，先运行 `uv python install 3.12`。

```bash
uv sync                       # 根据项目配置和锁文件创建/同步 .venv
uv run python main.py         # 在项目虚拟环境中运行当前入口
uv add <package>              # 添加运行时依赖
uv add --dev <package>        # 添加开发依赖
uv remove <package>           # 移除依赖
```

项目还定义了 `rag-agent-playground` 命令行入口，指向 `rag_agent_playground:main`。当前尚未配置测试、Lint 或格式化工具；后续如引入这些工具，请在本文件中补充对应命令。

## 项目结构

- `main.py`：当前最小化的顶层运行入口。
- `src/rag_agent_playground/`：Python 项目包。
- `pyproject.toml`：项目元数据和依赖声明。
- `uv.lock`：锁定解析后的依赖版本。
- `.python-version`：指定项目使用的 Python 版本。

新增应用模块应放在 `src/rag_agent_playground/` 下，并遵循现有包结构。用户可见的安装步骤、配置或行为发生变化时，请同步更新 `README.md`。

## 协作约定

- 优先提交符合项目阶段、范围清晰的改动；不要将规划中的功能描述为已实现。
- 文档和面向用户的说明以中文为主；Python 模块、类、函数和变量使用清晰易懂的英文名称。
- 仅在有助于说明不明显的行为或公共接口时添加注释和文档字符串，避免重复代码含义的注释。
- 通过 `uv` 管理依赖，依赖声明写入 `pyproject.toml`，并提交对应的 `uv.lock` 更新。
- API Key 等敏感信息从环境变量读取。不要提交凭据或真实密钥。新增必需的环境变量时，在 `README.md` 中说明变量名及用途；`.env` 文件已由 Git 忽略。
- 仓库尚未声明许可证；不要在代码或文档中擅自假设项目采用某种许可证。
