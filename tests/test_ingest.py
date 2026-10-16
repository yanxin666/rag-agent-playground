"""文档加载、解析与分块测试（离线，不依赖网络）。"""

from __future__ import annotations

import pytest

from rag_agent.rag.ingest import load_chunks


def _write_doc(tmp_path, name: str, text: str) -> None:
    """在临时目录里写一个文本文件。"""
    (tmp_path / name).write_text(text, encoding="utf-8")


def test_load_and_chunk_single_doc(tmp_path):
    """单个长文档能被切成多个非空文本块。"""
    _write_doc(
        tmp_path,
        "note.md",
        "# 标题\n\n" + "这是一段用于测试分块的中文内容。" * 30,
    )
    chunks = load_chunks(tmp_path, chunk_size=64, chunk_overlap=8)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) and c for c in chunks)
    assert "测试分块" in "".join(chunks)


def test_chunk_params_affect_result(tmp_path):
    """更小的 chunk_size 会产生更多文本块。"""
    _write_doc(tmp_path, "note.md", "句子一。句子二。句子三。句子四。句子五。句子六。句子七。" * 10)
    small = load_chunks(tmp_path, chunk_size=128, chunk_overlap=10)
    large = load_chunks(tmp_path, chunk_size=1024, chunk_overlap=64)
    assert len(small) > len(large)


def test_multiple_files_and_extension_filter(tmp_path):
    """目录下多个文档都被加载，required_exts 能按扩展名过滤。"""
    _write_doc(tmp_path, "a.md", "# 文档A\n\n内容甲。")
    _write_doc(tmp_path, "b.md", "# 文档B\n\n内容乙。")
    _write_doc(tmp_path, "ignore.txt", "不应被加载")
    all_chunks = load_chunks(tmp_path, chunk_size=128, chunk_overlap=10)
    assert "内容甲" in "".join(all_chunks)
    assert "内容乙" in "".join(all_chunks)
    md_only = load_chunks(tmp_path, chunk_size=128, chunk_overlap=10, required_exts=[".md"])
    assert "不应被加载" not in "".join(md_only)


def test_empty_dir_raises(tmp_path):
    """目录里没有任何可解析文件时抛 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path, chunk_size=128, chunk_overlap=10)


def test_sample_docs_loadable():
    """仓库自带的端到端样例文档 data/sample_docs 可正常加载分块。"""
    chunks = load_chunks("data/sample_docs", chunk_size=256, chunk_overlap=20)
    assert len(chunks) >= 3
    assert "RAG" in "".join(chunks)
