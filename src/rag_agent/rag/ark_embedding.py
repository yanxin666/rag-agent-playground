"""火山方舟（豆包）文本向量化：直接通过 HTTP 调用 embeddings API。"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

ARK_EMBEDDINGS_PATH = "/embeddings"  # 方舟 embeddings 接口路径
MAX_BATCH_SIZE = 16  # 方舟单次请求允许的最大文本条数


class ArkEmbedding(BaseEmbedding):
    """调用方舟 embeddings API 的 LlamaIndex 自定义 Embedding。

    豆包 embedding 系列不区分 query/document 向量化，query 与 text 走同一接口。
    """

    api_key: str
    base_url: str
    # PrivateAttr：pydantic 私有属性，不进构造参数、不参与序列化
    _client: httpx.Client = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """pydantic 初始化钩子：字段校验完成后创建复用的 HTTP 客户端。"""
        super().model_post_init(__context)
        self._client = httpx.Client(timeout=30.0)

    def _invoke_api(self, texts: list[str]) -> list[list[float]]:
        """调用方舟 /embeddings 接口，返回与输入文本顺序一一对应的向量列表。"""
        response = self._client.post(
            # rstrip("/") 兼容 base_url 带不带结尾斜杠两种写法
            f"{self.base_url.rstrip('/')}{ARK_EMBEDDINGS_PATH}",
            # Bearer token 鉴权
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model_name, "input": texts, "encoding_format": "float"},
        )
        if response.status_code != 200:
            # 带上响应正文片段，方便排查原因（key 无效、模型名错误等）
            raise RuntimeError(
                f"方舟 embeddings 调用失败: HTTP {response.status_code} {response.text[:500]}"
            )
        data = response.json()["data"]
        # 接口不保证按输入顺序返回，按 index 排序还原与输入的对应关系
        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]

    def _get_query_embedding(self, query: str) -> list[float]:
        """单条查询文本 -> 向量；由基类公开方法 get_query_embedding() 调用。"""
        return self._invoke_api([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """异步版查询向量化；本类只有同步 HTTP，直接复用同步实现。"""
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """单条文档文本 -> 向量；由基类公开方法 get_text_embedding() 调用。"""
        return self._invoke_api([text])[0]

    def get_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False, **kwargs: Any
    ) -> list[list[float]]:
        """按方舟单次请求上限分批调用，覆盖基类默认的逐条批处理。

        基类默认实现是一段文本发一次请求；这里改为每次最多 MAX_BATCH_SIZE 条，
        大幅减少文档入库时的请求次数。
        """
        embeddings: list[list[float]] = []
        # 从 start 开始切一批（最多 16 条），extend 拼接结果并保持原始顺序
        for start in range(0, len(texts), MAX_BATCH_SIZE):
            embeddings.extend(self._invoke_api(texts[start : start + MAX_BATCH_SIZE]))
        return embeddings
