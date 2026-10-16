"""config.load_settings 的单元测试。"""

from __future__ import annotations

import pytest

from rag_agent.config import (
    ARK_BASE_URL_DEFAULT,
    ARK_CHAT_MODEL_DEFAULT,
    ARK_EMBEDDING_MODEL_DEFAULT,
    EMBEDDING_DIM_DEFAULT,
    load_settings,
)


def test_defaults(no_env):
    """未配置任何环境变量时，全部落到代码内默认值。"""
    s = load_settings(no_env)
    assert s.ark_api_key == ""
    assert s.ark_base_url == ARK_BASE_URL_DEFAULT
    assert s.ark_chat_model == ARK_CHAT_MODEL_DEFAULT
    assert s.ark_embedding_model == ARK_EMBEDDING_MODEL_DEFAULT
    assert s.embedding_dim == EMBEDDING_DIM_DEFAULT
    assert s.milvus_uri == ""
    assert s.chunk_size == 512
    assert s.retrieval_top_k == 4


def test_reads_from_env(no_env, monkeypatch):
    """进程环境变量中的配置能被正确读到。"""
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    monkeypatch.setenv("MILVUS_URI", "https://example.zilliz.com")
    monkeypatch.setenv("MILVUS_TOKEN", "tok")
    monkeypatch.setenv("CHUNK_SIZE", "256")
    s = load_settings(no_env)
    assert s.ark_api_key == "test-key"
    assert s.milvus_uri == "https://example.zilliz.com"
    assert s.milvus_token == "tok"
    assert s.chunk_size == 256


def test_reads_from_env_file(no_env, tmp_path):
    """.env 文件中的配置能被正确读到。"""
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "ARK_API_KEY=file-key\nARK_CHAT_MODEL=my-model\nCHUNK_SIZE=128\n",
        encoding="utf-8",
    )
    s = load_settings(env_file)
    assert s.ark_api_key == "file-key"
    assert s.ark_chat_model == "my-model"
    assert s.chunk_size == 128


def test_process_env_takes_priority_over_env_file(no_env, tmp_path, monkeypatch):
    """同名配置时，进程环境变量优先于 .env 文件。"""
    env_file = tmp_path / "test.env"
    env_file.write_text("ARK_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("ARK_API_KEY", "process-key")
    s = load_settings(env_file)
    assert s.ark_api_key == "process-key"


def test_require_helpers(no_env):
    """require_* 辅助方法在缺少必填配置时应抛 RuntimeError。"""
    s = load_settings(no_env)
    with pytest.raises(RuntimeError, match="ARK_API_KEY"):
        s.require_ark_api_key()
    with pytest.raises(RuntimeError, match="MILVUS"):
        s.require_milvus()
    assert load_settings(no_env).embedding_dim == EMBEDDING_DIM_DEFAULT
