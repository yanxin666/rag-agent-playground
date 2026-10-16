# 项目 FAQ

## 本项目的技术选型是什么？

本项目基于 Python 3.12，使用 LlamaIndex 做文档解析与检索，LangGraph 做智能体编排，向量库使用 Milvus（Zilliz 云北京区），包管理使用 uv。

## 默认的模型和参数是多少？

- Chat 模型：doubao-seed-2.1-pro（火山方舟）。
- Embedding 模型：doubao-embedding-text-240715，输出 2560 维 float32 向量。
- 分块参数：chunk_size=512，chunk_overlap=50。
- 检索参数：默认返回 top_k=4 个文本块。

## 知识库什么时候更新？

知识库每周三下午 4 点例行更新，紧急文档可随时通过 ingest 命令手动写入。

## 谁负责维护知识库？

知识库由平台工程组的"向量检索小队"负责维护，值班联系邮箱是 vector-team@example.com。

## 如何提交新的知识库文档？

把 Markdown 文档放入 data/sample_docs 目录，然后执行 `uv run python main.py ingest data/sample_docs`。
