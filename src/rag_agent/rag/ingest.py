"""文档加载、解析与分块流程：SimpleDirectoryReader + SentenceSplitter。"""

from __future__ import annotations

import logging
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)


def load_chunks(
    input_dir: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    required_exts: list[str] | None = None,
) -> list[str]:
    """加载目录下的文档并分块，返回文本块列表。

    required_exts 可限定只读取指定扩展名（如 [".md"]），None 表示读取所有支持类型。
    目录不存在或没有可解析文件时抛 FileNotFoundError。
    """
    # 前置校验目录存在，路径写错时尽早报错
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"目录 {input_dir} 不存在")
    # recursive=True：递归扫描子目录
    reader_kwargs: dict = {"input_dir": str(input_path), "recursive": True}
    if required_exts:
        reader_kwargs["required_exts"] = required_exts
    try:
        # SimpleDirectoryReader 按扩展名自动选择解析器（md/txt/pdf 等）
        documents = SimpleDirectoryReader(**reader_kwargs).load_data()
    except ValueError as exc:
        # llama_index 对"没有任何可解析文件"抛 ValueError，这里转成更准确的 FileNotFoundError
        raise FileNotFoundError(f"目录 {input_dir} 下没有可解析的文档") from exc
    logger.info("已加载 %d 个文档", len(documents))

    # SentenceSplitter 优先在句子边界切分，尽量避免句子被拦腰截断
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Document -> Node（带元数据的文本块）；这里只取每个块的纯文本内容
    nodes = splitter.get_nodes_from_documents(documents)
    chunks = [node.get_content() for node in nodes]
    logger.info(
        "已分块为 %d 个文本块 (chunk_size=%d, chunk_overlap=%d)",
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
