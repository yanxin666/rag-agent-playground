"""MilvusRetriever 单元测试。"""

from __future__ import annotations

from typing import Any

import pytest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.schema import QueryBundle

from rag_agent.rag.milvus_store import MilvusDocumentStore
from rag_agent.rag.retriever import MilvusRetriever


class RecordingClient:
    """记录 search 调用参数并返回固定命中结果。"""

    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits
        self.search_kwargs: dict | None = None

    def has_collection(self, collection_name: str, **kwargs) -> bool:
        return True

    def create_collection(self, collection_name: str, **kwargs) -> None:
        raise AssertionError("不应被调用")

    def insert(self, collection_name: str, data: Any, **kwargs) -> dict:
        raise AssertionError("不应被调用")

    def search(
        self,
        collection_name: str,
        data: Any = None,
        filter: str = "",
        limit: int = 10,
        output_fields: list[str] | None = None,
        **kwargs,
    ) -> list[list[dict]]:
        self.search_kwargs = {
            "collection_name": collection_name,
            "data": data,
            "limit": limit,
            "output_fields": output_fields,
        }
        return [self.hits]

    def drop_collection(self, collection_name: str, **kwargs) -> None:
        raise AssertionError("不应被调用")


def make_retriever(hits: list[dict], dim: int = 4, top_k: int = 3):
    """用 RecordingClient 和 MockEmbedding（固定维度的假向量）组装待测 Retriever。"""
    client = RecordingClient(hits)
    store = MilvusDocumentStore(client, "c", dim)
    retriever = MilvusRetriever(store=store, embed_model=MockEmbedding(embed_dim=dim), top_k=top_k)
    return retriever, client


def test_retrieve_converts_hits_to_scored_nodes():
    """命中结果被转成带分数的节点，distance 换算为相似度分。"""
    hits = [
        {"id": 1, "distance": 0.25, "entity": {"text": "第一段"}},
        {"id": 2, "distance": 0.75, "entity": {"text": "第二段"}},
    ]
    retriever, _ = make_retriever(hits)
    nodes = retriever.retrieve(QueryBundle("查询词"))
    assert [n.node.get_content() for n in nodes] == ["第一段", "第二段"]
    assert nodes[0].score == pytest.approx(0.75)
    assert nodes[1].score == pytest.approx(0.25)


def test_retrieve_passes_embedding_and_top_k():
    """检索时正确传入查询向量的维度与 top_k。"""
    retriever, client = make_retriever([], dim=4, top_k=5)
    nodes = retriever.retrieve(QueryBundle("查询词"))
    assert nodes == []
    assert client.search_kwargs is not None
    kwargs = client.search_kwargs
    assert kwargs["limit"] == 5
    assert kwargs["output_fields"] == ["text"]
    assert len(kwargs["data"][0]) == 4  # MockEmbedding 输出维度
