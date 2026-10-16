# Milvus 与 Zilliz Cloud 使用要点

Milvus 是开源向量数据库，Zilliz Cloud 是其官方全托管云服务，本项目使用 Zilliz 云北京区。

## 核心概念

- **集合（Collection）**：类似关系库中的表，存放向量与标量字段。
- **维度（Dimension）**：向量维度必须与 Embedding 模型输出一致，本项目豆包 embedding 输出 2560 维 float32 向量。
- **相似度度量**：常用 COSINE（余弦相似度）、L2（欧氏距离）、IP（内积）。本项目统一使用 COSINE。
- **快速建集合**：通过 MilvusClient 的 create_collection 传入维度即可自动创建带默认索引的集合。

## 本项目的存取约定

- 写入字段：`vector`（向量）、`text`（原文块）、`chunk_index`（块序号），id 由 Milvus 自动生成（auto_id）。
- 检索：输入查询向量与 top_k，返回按相似度排序的文本块。
- 连接：通过 `MILVUS_URI` 与 `MILVUS_TOKEN` 配置 Zilliz 云端点与 API Key。

## 运维提示

- 集合维度建好后不可修改，更换 Embedding 模型需要重建集合并重新写入。
- 定期清理不再使用的集合，避免云端存储浪费。
