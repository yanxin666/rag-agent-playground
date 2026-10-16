"""Milvus 向量检索封装为 LlamaIndex Retriever，并组装 QueryEngine。"""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.llms.openai_like import OpenAILike
from pydantic import PrivateAttr

from rag_agent.config import Settings
from rag_agent.rag.milvus_store import MilvusDocumentStore

logger = logging.getLogger(__name__)


class MilvusRetriever(BaseRetriever):
    """查询文本 -> 向量 -> Milvus 检索 -> NodeWithScore 列表。"""

    # 三个私有依赖：存储（负责检索）、embedding 模型（负责向量化）、top_k（返回条数）
    _store: MilvusDocumentStore = PrivateAttr()
    _embed_model: BaseEmbedding = PrivateAttr()
    _top_k: int = PrivateAttr()

    def __init__(
        self,
        store: MilvusDocumentStore,
        embed_model: BaseEmbedding,
        top_k: int = 4,
        **kwargs: Any,
    ) -> None:
        """保存依赖；额外的 kwargs 透传给 BaseRetriever 基类。"""
        super().__init__(**kwargs)
        self._store = store
        self._embed_model = embed_model
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """检索核心逻辑：问题文本 -> 向量 -> Milvus 相似检索 -> 节点列表。

        基类公开方法 retrieve() 内部会调用本方法。
        """
        # 1. 把问题文本转成向量（查询与入库走同一 embedding 模型，向量空间才一致）
        embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        # 2. 用向量在 Milvus 中检索最相似的 top_k 个文本块
        hits = self._store.search(embedding, self._top_k)
        logger.debug("检索到 %d 条结果", len(hits))
        # 3. 把命中结果包装成 LlamaIndex 的 NodeWithScore
        nodes = []
        for index, hit in enumerate(hits):
            node = TextNode(text=hit["text"], id_=f"milvus-hit-{index}")
            # COSINE 度量下 distance 越小越相似；转成"分数越大越好"的 score
            nodes.append(NodeWithScore(node=node, score=1.0 - hit["distance"]))
        return nodes


def build_query_engine(settings: Settings) -> RetrieverQueryEngine:
    """组装完整检索链路：Ark embedding + Milvus 检索 + 豆包 LLM 综合回答。"""
    from rag_agent.rag.ark_embedding import ArkEmbedding

    # 1. 连接 Milvus
    uri, token = settings.require_milvus()
    store = MilvusDocumentStore.connect(
        uri=uri, token=token, collection=settings.milvus_collection, dim=settings.embedding_dim
    )
    # 2. 构造向量化模型
    embed_model = ArkEmbedding(
        api_key=settings.require_ark_api_key(),
        base_url=settings.ark_base_url,
        model_name=settings.ark_embedding_model,
    )
    # 3. Retriever 负责"取材"：从 Milvus 找出与问题相关的文本块
    retriever = MilvusRetriever(
        store=store, embed_model=embed_model, top_k=settings.retrieval_top_k
    )
    # 4. LLM 负责"作答"：OpenAILike 走 OpenAI 兼容协议接入豆包
    llm = OpenAILike(
        model=settings.ark_chat_model,
        api_base=settings.ark_base_url,
        api_key=settings.require_ark_api_key(),
        is_chat_model=True,
    )
    # 5. COMPACT 模式：把检索到的文本块压缩后交给 LLM 生成一段回答
    synthesizer = get_response_synthesizer(
        llm=llm, response_mode=ResponseMode.COMPACT, streaming=False
    )
    # QueryEngine = Retriever（检索）+ ResponseSynthesizer（综合回答）
    return RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)
