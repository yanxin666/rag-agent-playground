"""命令行入口：ingest 写入知识库，ask 向 Agent 提问。"""

from __future__ import annotations

import argparse

from rag_agent.config import Settings, load_settings
from rag_agent.log import setup_logging


def run_ingest(settings: Settings, input_dir: str, rebuild: bool) -> None:
    """执行 ingest 子命令：读取目录文档 -> 分块 -> 向量化 -> 写入 Milvus。

    各板块模块在此延迟导入，只有真正执行 ingest 时才加载这些重依赖。
    """
    from rag_agent.rag.ark_embedding import ArkEmbedding
    from rag_agent.rag.ingest import load_chunks
    from rag_agent.rag.milvus_store import MilvusDocumentStore

    # 1. 读取目录下所有文档，按 chunk_size / chunk_overlap 切成文本块
    chunks = load_chunks(input_dir, settings.chunk_size, settings.chunk_overlap)
    # 2. 连接 Milvus；--rebuild 时先删旧集合，保证从零重建
    uri, token = settings.require_milvus()
    store = MilvusDocumentStore.connect(
        uri=uri,
        token=token,
        collection=settings.milvus_collection,
        dim=settings.embedding_dim,
    )
    if rebuild:
        store.drop_collection()
    store.ensure_collection()
    # 3. 调方舟 embedding 接口，把所有文本块批量转成向量
    embed_model = ArkEmbedding(
        api_key=settings.require_ark_api_key(),
        base_url=settings.ark_base_url,
        model_name=settings.ark_embedding_model,
    )
    embeddings = embed_model.get_text_embedding_batch(chunks)
    # 4. 文本块与对应向量一起写入 Milvus
    count = store.insert_chunks(chunks, embeddings)
    print(f"已写入 {count} 个文本块到集合 {settings.milvus_collection}")


def run_ask(settings: Settings, question: str) -> None:
    """执行 ask 子命令：组装 RAG 检索链路与 LangGraph Agent，回答一个问题。"""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    from rag_agent.agent.graph import build_agent_graph
    from rag_agent.agent.tools import make_rag_search_tool
    from rag_agent.rag.retriever import build_query_engine

    # 1. 组装 QueryEngine：embedding + Milvus 检索 + LLM 综合回答
    query_engine = build_query_engine(settings)
    # 2. 用 OpenAI 兼容 SDK 接入豆包 Chat 模型；SecretStr 标记密钥，避免被日志/异常打印泄露
    llm = ChatOpenAI(
        model=settings.ark_chat_model,
        api_key=SecretStr(settings.require_ark_api_key()),
        base_url=settings.ark_base_url,
    )
    # 3. 构建带 rag_search 工具的 Agent 状态图，携带用户问题启动执行
    graph = build_agent_graph(llm, [make_rag_search_tool(query_engine)])
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    # Agent 循环结束后，最后一条 AI 消息即最终回答
    print(result["messages"][-1].content)


def main(argv: list[str] | None = None) -> None:
    """CLI 入口：解析 ingest / ask 子命令并分发到对应的 run_* 函数。

    argv 为 None 时解析真实命令行参数；测试时可显式传入参数列表。
    """
    parser = argparse.ArgumentParser(
        prog="rag-agent-playground", description="RAG + LangGraph 知识库问答"
    )
    # required=True：必须给出子命令，否则打印用法并以退出码 2 结束
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="加载、分块并向量化文档，写入 Milvus")
    p_ingest.add_argument("input_dir", help="文档目录路径，如 data/sample_docs")
    p_ingest.add_argument("--rebuild", action="store_true", help="先删除已有集合再重建")

    p_ask = sub.add_parser("ask", help="向知识库 Agent 提问")
    p_ask.add_argument("question", help="问题文本")

    args = parser.parse_args(argv)

    # 先统一加载配置、初始化日志，再按子命令分发
    settings = load_settings()
    setup_logging(settings.log_level)
    if args.command == "ingest":
        run_ingest(settings, args.input_dir, args.rebuild)
    elif args.command == "ask":
        run_ask(settings, args.question)
