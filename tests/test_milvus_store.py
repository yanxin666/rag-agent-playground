"""MilvusDocumentStore 单元测试：用内存版 FakeMilvusClient 模拟 Milvus 行为。"""

from __future__ import annotations

import math

import pytest

from rag_agent.rag.milvus_store import MilvusDocumentStore


class FakeMilvusClient:
    """实现本项目用到的 MilvusClient 接口子集，检索按余弦相似度模拟。"""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.collections: dict[str, dict] = {}

    def has_collection(self, collection_name: str, **kwargs) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, **kwargs) -> None:
        self.collections[collection_name] = kwargs

    def insert(self, collection_name: str, data: list[dict], **kwargs) -> dict:
        self.rows.extend(data)
        return {"insert_count": len(data), "ids": list(range(len(data)))}

    def search(
        self,
        collection_name: str,
        data: list | None = None,
        filter: str = "",
        limit: int = 10,
        output_fields: list[str] | None = None,
        **kwargs,
    ) -> list[list[dict]]:
        assert data is not None
        query = data[0]

        def cosine(a: list[float], b: list[float]) -> float:
            # 标准余弦相似度：点积 / (两个向量模长之积)，越大越相似
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
            return dot / norm

        # 按相似度从高到低排序，取前 limit 条
        ranked = sorted(self.rows, key=lambda row: -cosine(query, row["vector"]))[:limit]
        return [
            [
                {
                    "id": i,
                    "distance": 1 - cosine(query, row["vector"]),
                    "entity": {"text": row["text"]},
                }
                for i, row in enumerate(ranked)
            ]
        ]

    def drop_collection(self, collection_name: str, **kwargs) -> None:
        self.collections.pop(collection_name, None)


def make_store() -> tuple[MilvusDocumentStore, FakeMilvusClient]:
    """构造 store + 内存版 client，返回两者便于断言 client 侧状态。"""
    client = FakeMilvusClient()
    return MilvusDocumentStore(client, "test_collection", dim=3), client


def test_ensure_collection_creates_with_dim_and_auto_id():
    """建表参数（维度、主键、度量类型）正确，且重复调用不会重复建表。"""
    store, client = make_store()
    store.ensure_collection()
    assert "test_collection" in client.collections
    kwargs = client.collections["test_collection"]
    assert kwargs["dimension"] == 3
    assert kwargs["auto_id"] is True
    assert kwargs["metric_type"] == "COSINE"
    store.ensure_collection()  # 幂等，不重复创建
    assert list(client.collections) == ["test_collection"]


def test_insert_then_search_roundtrip():
    """写入后能按向量相似度检索回最相关的文本。"""
    store, _ = make_store()
    store.ensure_collection()
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]]
    count = store.insert_chunks(["红色文档", "蓝色文档", "偏红文档"], vectors)
    assert count == 3

    hits = store.search([1.0, 0.0, 0.0], top_k=2)
    assert len(hits) == 2
    assert hits[0]["text"] == "红色文档"
    assert hits[0]["distance"] <= hits[1]["distance"]
    assert hits[1]["text"] == "偏红文档"


def test_insert_mismatch_raises():
    """chunks 与 embeddings 数量不一致时抛 ValueError。"""
    store, _ = make_store()
    store.ensure_collection()
    with pytest.raises(ValueError, match="数量不一致"):
        store.insert_chunks(["只有文本"], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


def test_drop_collection():
    """删除集合幂等：集合不存在时再删也不报错。"""
    store, client = make_store()
    store.ensure_collection()
    store.drop_collection()
    assert "test_collection" not in client.collections
    store.drop_collection()  # 不存在时静默通过
