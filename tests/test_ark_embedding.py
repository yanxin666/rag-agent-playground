"""ArkEmbedding 单元测试：用 httpx.MockTransport 模拟方舟 API。"""

from __future__ import annotations

import json

import httpx
import pytest

from rag_agent.rag.ark_embedding import (
    ARK_EMBEDDINGS_PATH,
    MAX_BATCH_SIZE,
    ArkEmbedding,
)


def make_embedding(handler) -> ArkEmbedding:
    """构造测试实例：用 MockTransport 把 HTTP 请求拦截到 handler，不发真实网络请求。"""
    embedding = ArkEmbedding(
        api_key="test-key",
        base_url="https://ark.test/api/v3/",
        model_name="test-model",
    )
    embedding._client = httpx.Client(transport=httpx.MockTransport(handler))
    return embedding


def ok_response(count: int) -> httpx.Response:
    """生成包含 count 条向量的成功响应。"""
    return httpx.Response(
        200,
        json={"data": [{"index": i, "embedding": [float(i), 1.0]} for i in range(count)]},
    )


def test_query_embedding_request_and_parsing():
    """请求 URL、鉴权头、请求体正确；返回结果按 index 还原输入顺序。"""
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.2, 0.2]},
                    {"index": 0, "embedding": [0.1, 0.1]},
                ]
            },
        )

    embedding = make_embedding(handler)
    assert embedding.get_query_embedding("你好") == [0.1, 0.1]  # 按 index 排序

    request = requests[0]
    assert str(request.url) == f"https://ark.test/api/v3{ARK_EMBEDDINGS_PATH}"
    assert request.headers["Authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body == {"model": "test-model", "input": ["你好"], "encoding_format": "float"}


def test_batch_embedding_splits_by_max_batch_size():
    """批量接口按 MAX_BATCH_SIZE 自动分批请求。"""
    bodies = []
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        data = []
        for _ in body["input"]:
            data.append({"index": len(data), "embedding": [float(counter["n"]), 1.0]})
            counter["n"] += 1
        return httpx.Response(200, json={"data": data})

    embedding = make_embedding(handler)
    count = MAX_BATCH_SIZE + 4
    texts = [f"文本{i}" for i in range(count)]
    result = embedding.get_text_embedding_batch(texts)

    assert len(result) == count
    assert [len(b["input"]) for b in bodies] == [MAX_BATCH_SIZE, 4]
    assert result[MAX_BATCH_SIZE] == [float(MAX_BATCH_SIZE), 1.0]


def test_http_error_raises_runtime_error():
    """HTTP 非 200 时抛出带状态码的 RuntimeError。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid api key")

    embedding = make_embedding(handler)
    with pytest.raises(RuntimeError, match="401"):
        embedding.get_text_embedding("任意文本")


def test_async_query_embedding_delegates():
    """异步接口委托到同步实现，结果一致。"""
    import asyncio

    def handler(request: httpx.Request) -> httpx.Response:
        return ok_response(1)

    embedding = make_embedding(handler)
    assert asyncio.run(embedding.aget_query_embedding("你好")) == [0.0, 1.0]
