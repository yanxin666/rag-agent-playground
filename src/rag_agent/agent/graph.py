"""LangGraph Agent 状态图：agent 节点 -> 条件路由 -> ToolNode 循环。"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 系统提示词：约束 Agent 行为——涉及知识库的问题先检索再回答
SYSTEM_PROMPT = (
    "你是一个知识库问答助手。收到用户问题后，如果需要查阅资料，"
    "请先调用 rag_search 工具检索知识库，再依据检索结果回答；"
    "如果问题与知识库无关或无需检索，可以直接回答。回答请使用中文。"
)


class AgentState(TypedDict):
    """Agent 的共享状态：所有节点读写同一个 messages 列表。"""

    # Annotated + add_messages：节点返回的新消息由 add_messages 自动"追加/按 id 更新"，
    # 而不是整体覆盖旧消息，从而保留完整对话历史
    messages: Annotated[list, add_messages]


def route_after_agent(state: AgentState) -> str:
    """有工具调用请求则进入 tools 节点，否则结束。"""
    # 只看最新一条 AI 消息是否携带工具调用请求
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_agent_graph(llm: BaseChatModel, tools: list[BaseTool]):
    """构建并编译 Agent 状态图。

    START -> agent -> (有 tool_calls) -> tools -> agent 循环
                  -> (无 tool_calls) -> END
    """
    # 把工具清单（名称 + 参数 schema）附加给 LLM，LLM 才知道可以调用哪些工具
    llm_with_tools = llm.bind_tools(tools)

    def call_agent(state: AgentState):
        """调用 LLM 生成下一条消息：system 提示词 + 当前完整对话历史。"""
        response = llm_with_tools.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]]
        )
        return {"messages": [response]}  # 返回值经 add_messages 合并进状态

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent)  # agent 节点：调用 LLM，决定直接回复还是发起工具调用
    graph.add_node("tools", ToolNode(tools))  # tools 节点：真正执行工具，结果以 ToolMessage 回传
    graph.add_edge(START, "agent")  # 入口固定先进入 agent
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    # agent 之后按"是否发起工具调用"分流（见 route_after_agent）
    graph.add_edge("tools", "agent")  # 工具执行完回到 agent，形成"调用-观察"循环
    return graph.compile()  # 编译成可 invoke 的可执行图
