"""基于官方 pymilvus MilvusClient（v2 API）的向量存取薄封装，连接 Zilliz 云。"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from pymilvus import MilvusClient

logger = logging.getLogger(__name__)


class MilvusClientLike(Protocol):
    """本项目用到的 MilvusClient 接口子集，便于测试替换。

    Protocol 是"结构化类型"（鸭子类型的静态写法）：任何实现了这些方法签名的类
    都算满足本协议，测试里的 FakeMilvusClient 无需继承真实 MilvusClient 即可注入。
    pymilvus 3.0.2 把 has_collection 的返回类型误标为 Coroutine（运行时实际是同步 bool），
    因此协议中放宽为 Any。
    """

    def has_collection(self, collection_name: str, **kwargs: Any) -> Any: ...

    def create_collection(self, collection_name: str, **kwargs: Any) -> None: ...

    def insert(self, collection_name: str, data: Any, **kwargs: Any) -> dict: ...

    def search(
        self,
        collection_name: str,
        data: Any = None,
        filter: str = "",
        limit: int = 10,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[list[dict]]: ...

    def drop_collection(self, collection_name: str, **kwargs: Any) -> None: ...


class MilvusDocumentStore:
    """Milvus 集合的薄封装：建表、写入、向量检索、删表。

    client 通过构造函数注入（依赖注入），测试时可传入内存版替身而不连真实服务。
    """

    def __init__(self, client: MilvusClientLike, collection: str, dim: int) -> None:
        """client：真实 MilvusClient 或测试替身；collection：集合名；dim：向量维度。"""
        self.client = client
        self.collection = collection
        self.dim = dim

    @classmethod
    def connect(cls, uri: str, token: str, collection: str, dim: int) -> MilvusDocumentStore:
        """用地址 + token 连接 Zilliz 云上的 Milvus，返回封装好的 store 实例。"""
        return cls(MilvusClient(uri=uri, token=token), collection, dim)

    def ensure_collection(self) -> None:
        """确保集合存在；可重复调用（幂等），已存在时直接跳过。"""
        if self.client.has_collection(self.collection):
            return
        # auto_id=True：主键由 Milvus 自动生成；metric_type="COSINE"：按余弦相似度检索
        self.client.create_collection(
            collection_name=self.collection,
            dimension=self.dim,
            metric_type="COSINE",
            auto_id=True,
        )
        logger.info("已创建 Milvus 集合 %s (dim=%d)", self.collection, self.dim)

    def insert_chunks(self, chunks: list[str], embeddings: list[list[float]]) -> int:
        """插入文本块与对应向量，文本存入 text 字段（动态字段），块序号存入 chunk_index。

        返回成功写入的条数。
        """
        # 文本块与向量必须一一对应，先做数量校验
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks({len(chunks)}) 与 embeddings({len(embeddings)}) 数量不一致")
        # 每行一条记录：vector 用于检索，text/chunk_index 存原文与序号便于取回和排查
        rows = [
            {"vector": embedding, "text": chunk, "chunk_index": index}
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        result = self.client.insert(collection_name=self.collection, data=rows)
        # insert_count 为服务端返回的成功条数；响应格式异常时按本地行数兜底
        count = result.get("insert_count", len(rows)) if isinstance(result, dict) else len(rows)
        logger.info("已写入 %d 个文本块到集合 %s", count, self.collection)
        return count

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """向量检索，返回 [{text, distance}]，按相似度降序。"""
        # data 是"查询向量列表"；这里只传 1 个查询向量，因此外层结果也只有一组
        results = self.client.search(
            collection_name=self.collection,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text"],  # 命中时额外带回 text 字段（原文）
        )
        hits: list[dict[str, Any]] = []
        # 双层结构：外层对应每个查询向量，内层是该查询的命中记录
        for result in results:
            for hit in result:
                hits.append({"text": hit["entity"]["text"], "distance": hit["distance"]})
        return hits

    def drop_collection(self) -> None:
        """删除集合；集合不存在时静默跳过（幂等）。"""
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
            logger.info("已删除集合 %s", self.collection)
