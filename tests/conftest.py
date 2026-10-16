"""共享测试夹具。"""

from __future__ import annotations

import pytest

ENV_KEYS = [
    "ARK_API_KEY",
    "ARK_BASE_URL",
    "ARK_CHAT_MODEL",
    "ARK_EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "MILVUS_URI",
    "MILVUS_TOKEN",
    "MILVUS_COLLECTION",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVAL_TOP_K",
    "LOG_LEVEL",
]


@pytest.fixture
def no_env(monkeypatch, tmp_path):
    """清空相关环境变量，并将 dotenv 指向不存在的文件，保证测试隔离。"""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return str(tmp_path / "absent.env")
