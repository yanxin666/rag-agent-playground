"""rag_search 工具单元测试。"""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from rag_agent.agent.tools import SNIPPET_MAX_CHARS, make_rag_search_tool


class FakeQueryResponse:
    """模拟 LlamaIndex 查询响应：str() 即综合回答，source_nodes 是引用片段。"""

    def __init__(self, answer: str, source_texts: list[str]) -> None:
        self._answer = answer
        self.source_nodes = [NodeWithScore(node=TextNode(text=text)) for text in source_texts]

    def __str__(self) -> str:
        return self._answer


class FakeQueryEngine:
    """记录每次查询并返回固定答案的 QueryEngine 替身。"""

    def __init__(self, answer: str = "综合回答", source_texts: list[str] | None = None) -> None:
        self.answer = answer
        self.source_texts = ["片段一", "片段二"] if source_texts is None else source_texts
        self.queries: list[str] = []

    def query(self, query: str) -> FakeQueryResponse:
        self.queries.append(query)
        return FakeQueryResponse(self.answer, self.source_texts)


def test_tool_returns_answer_and_snippets():
    """工具返回"综合回答 + 原文片段"两部分内容。"""
    engine = FakeQueryEngine()
    tool = make_rag_search_tool(engine)
    result = tool.invoke({"query": "什么是 RAG"})
    assert engine.queries == ["什么是 RAG"]
    assert "综合回答" in result
    assert "片段一" in result
    assert "片段二" in result
    assert "相关原文片段" in result


def test_snippets_truncated():
    """超长片段被截断到 SNIPPET_MAX_CHARS 以内。"""
    long_text = "长" * (SNIPPET_MAX_CHARS + 100)
    engine = FakeQueryEngine(source_texts=[long_text])
    tool = make_rag_search_tool(engine)
    result = tool.invoke({"query": "q"})
    snippet_line = [line for line in result.splitlines() if line.startswith("- ")][0]
    assert len(snippet_line) <= SNIPPET_MAX_CHARS + 2


def test_no_source_nodes_still_returns_answer():
    """没有 source_nodes 时仅返回综合回答。"""
    engine = FakeQueryEngine(source_texts=[])
    tool = make_rag_search_tool(engine)
    result = tool.invoke({"query": "q"})
    assert result.strip() == "综合回答"


def test_tool_metadata():
    """工具名与工具描述（发给 LLM 的元信息）正确。"""
    tool = make_rag_search_tool(FakeQueryEngine())
    assert tool.name == "rag_search"
    assert tool.description
