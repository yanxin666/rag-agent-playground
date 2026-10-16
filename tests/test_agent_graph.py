"""LangGraph Agent 状态图测试：验证工具调用循环与结果回传。"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_agent.agent.graph import AgentState, build_agent_graph, route_after_agent
from rag_agent.agent.tools import make_rag_search_tool


class FakeQueryResponse:
    """模拟查询响应：str() 即答案，不带 source_nodes。"""

    source_nodes: list = []

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def __str__(self) -> str:
        return self._answer


class FakeQueryEngine:
    """记录查询历史的 QueryEngine 替身。"""

    def __init__(self, answer: str = "知识库说：每周三下午更新") -> None:
        self.answer = answer
        self.queries: list[str] = []

    def query(self, query: str) -> FakeQueryResponse:
        self.queries.append(query)
        return FakeQueryResponse(self.answer)


class ToolBindableFakeChatModel(FakeMessagesListChatModel):
    """BaseChatModel.bind_tools 默认抛 NotImplementedError，测试替身返回自身即可。"""

    def bind_tools(self, tools, **kwargs):
        return self


def make_tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    """构造一条带工具调用请求的 AIMessage（模拟 LLM 决定调用工具）。"""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_agent_calls_tool_then_answers():
    """Agent 先调用 rag_search 检索，再基于工具结果回答（完整循环）。"""
    engine = FakeQueryEngine()
    tool = make_rag_search_tool(engine)
    llm = ToolBindableFakeChatModel(
        responses=[
            make_tool_call(
                "rag_search", {"query": "知识库更新时间"}, "call_1"
            ),  # 第 1 轮：请求调用工具
            AIMessage(content="知识库每周三下午更新。"),  # 第 2 轮：拿到工具结果后作答
        ]
    )
    graph = build_agent_graph(llm, [tool])

    result = graph.invoke({"messages": [HumanMessage(content="知识库什么时候更新？")]})
    messages = result["messages"]

    assert len(messages) == 4
    assert isinstance(messages[1], AIMessage) and messages[1].tool_calls
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].tool_call_id == "call_1"
    assert "每周三下午" in messages[2].content  # 工具执行结果已回传
    assert engine.queries == ["知识库更新时间"]
    assert messages[3].content == "知识库每周三下午更新。"


def test_agent_direct_answer_without_tool():
    """无需检索时 Agent 直接回答，不触发工具。"""
    engine = FakeQueryEngine()
    llm = ToolBindableFakeChatModel(responses=[AIMessage(content="直接回答，无需检索。")])
    graph = build_agent_graph(llm, [make_rag_search_tool(engine)])

    result = graph.invoke({"messages": [HumanMessage(content="你好")]})
    assert len(result["messages"]) == 2
    assert engine.queries == []  # 未触发工具
    assert result["messages"][-1].content == "直接回答，无需检索。"


def test_route_after_agent():
    """路由函数：最新消息带工具调用去 tools，否则结束。"""
    with_tool: AgentState = {"messages": [make_tool_call("rag_search", {"query": "x"}, "c")]}
    assert route_after_agent(with_tool) == "tools"
    without_tool: AgentState = {"messages": [AIMessage(content="回答")]}
    assert route_after_agent(without_tool) == "__end__"
