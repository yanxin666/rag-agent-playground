"""暴露给 LangGraph Agent 的工具：基于 QueryEngine 的知识库检索。"""

from __future__ import annotations

from typing import Any, Protocol

from langchain_core.tools import BaseTool, tool

SNIPPET_MAX_CHARS = 300  # 每条原文片段的截断长度，控制返回给 Agent 的内容体积


class QueryEngineLike(Protocol):
    """QueryEngine 的最小接口，便于测试替换（LlamaIndex QueryEngine 均满足）。"""

    def query(self, query: str) -> Any: ...


def make_rag_search_tool(query_engine: QueryEngineLike) -> BaseTool:
    """把 QueryEngine 包装成 LangChain 工具；内部函数通过闭包持有 engine 引用。"""

    # @tool 装饰器：把函数注册为工具——函数名即工具名，docstring 即发给 LLM 的工具描述，
    # 因此该 docstring 需保持对模型清晰的表述，不要改写成实现细节
    @tool
    def rag_search(query: str) -> str:
        """在知识库中检索与问题相关的文档内容，返回基于检索结果的回答及相关原文片段。"""
        # query_engine.query 内部完成：向量化 -> Milvus 检索 -> LLM 依据检索结果作答
        response = query_engine.query(query)
        # source_nodes 是回答所引用的原文片段，逐条整理成列表供 Agent 参考
        snippets = []
        for node in getattr(response, "source_nodes", []):
            # 换行压成空格并截断，避免单条片段过长
            snippet = node.get_content().strip().replace("\n", " ")[:SNIPPET_MAX_CHARS]
            snippets.append(f"- {snippet}")
        parts = [str(response).strip()]  # 综合回答放在最前
        if snippets:
            parts.append("相关原文片段:\n" + "\n".join(snippets))
        return "\n\n".join(parts)

    return rag_search
