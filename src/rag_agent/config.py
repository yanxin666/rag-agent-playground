"""集中配置管理：所有配置从项目根目录的 .env 文件或进程环境变量读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 各配置项的默认值集中在这里；load_settings() 读不到环境变量时逐项兜底
ARK_BASE_URL_DEFAULT = "https://ark.cn-beijing.volces.com/api/v3"
ARK_CHAT_MODEL_DEFAULT = "doubao-seed-2-1-pro-260628"
ARK_EMBEDDING_MODEL_DEFAULT = "doubao-embedding-text-240715"
EMBEDDING_DIM_DEFAULT = 2560
MILVUS_COLLECTION_DEFAULT = "rag_agent"
CHUNK_SIZE_DEFAULT = 512
CHUNK_OVERLAP_DEFAULT = 50
RETRIEVAL_TOP_K_DEFAULT = 4
LOG_LEVEL_DEFAULT = "INFO"


# frozen=True：实例创建后字段只读，防止配置在运行过程中被意外修改
@dataclass(frozen=True)
class Settings:
    """全局配置项集合，由 load_settings() 一次性构造，业务代码只读取。"""

    # —— 方舟（豆包）：API 密钥、服务地址与模型名 ——
    ark_api_key: str
    ark_base_url: str
    ark_chat_model: str
    ark_embedding_model: str
    embedding_dim: int  # embedding 向量维度，须与所选模型输出一致
    # —— Milvus（Zilliz 云）：连接信息与集合名 ——
    milvus_uri: str
    milvus_token: str
    milvus_collection: str
    # —— 文档分块与检索参数 ——
    chunk_size: int  # 每个文本块的目标长度（token 数）
    chunk_overlap: int  # 相邻文本块重叠的 token 数，避免关键句被切断
    retrieval_top_k: int  # 每次检索返回最相似的 k 个文本块
    # —— 日志 ——
    log_level: str

    def require_ark_api_key(self) -> str:
        """返回方舟 API Key；未配置时抛 RuntimeError（快速失败，避免空 key 去请求线上接口）。"""
        if not self.ark_api_key:
            raise RuntimeError("缺少 ARK_API_KEY，请在项目根目录 .env 中配置")
        return self.ark_api_key

    def require_milvus(self) -> tuple[str, str]:
        """返回 (uri, token) 二元组；任一缺失时抛 RuntimeError。"""
        if not self.milvus_uri or not self.milvus_token:
            raise RuntimeError(
                "缺少 MILVUS_URI / MILVUS_TOKEN，请在项目根目录 .env 中配置"
            )
        return self.milvus_uri, self.milvus_token


def load_settings(env_file: str | Path | None = None) -> Settings:
    """读取配置并构造 Settings。

    取值优先级：进程环境变量 > .env 文件 > 代码内默认值（上方 *_DEFAULT 常量）。
    env_file 为 None 时自动发现项目根目录的 .env。
    """
    # 把 .env 内容写入进程环境变量；已存在的进程变量不会被覆盖
    load_dotenv(env_file)
    # 每项用 os.environ.get(变量名, 默认值) 读取，读不到就落到默认值
    return Settings(
        ark_api_key=os.environ.get("ARK_API_KEY", ""),
        ark_base_url=os.environ.get("ARK_BASE_URL", ARK_BASE_URL_DEFAULT),
        ark_chat_model=os.environ.get("ARK_CHAT_MODEL", ARK_CHAT_MODEL_DEFAULT),
        ark_embedding_model=os.environ.get(
            "ARK_EMBEDDING_MODEL", ARK_EMBEDDING_MODEL_DEFAULT
        ),
        embedding_dim=int(os.environ.get("EMBEDDING_DIM", str(EMBEDDING_DIM_DEFAULT))),
        milvus_uri=os.environ.get("MILVUS_URI", ""),
        milvus_token=os.environ.get("MILVUS_TOKEN", ""),
        milvus_collection=os.environ.get(
            "MILVUS_COLLECTION", MILVUS_COLLECTION_DEFAULT
        ),
        chunk_size=int(os.environ.get("CHUNK_SIZE", str(CHUNK_SIZE_DEFAULT))),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", str(CHUNK_OVERLAP_DEFAULT))),
        retrieval_top_k=int(
            os.environ.get("RETRIEVAL_TOP_K", str(RETRIEVAL_TOP_K_DEFAULT))
        ),
        log_level=os.environ.get("LOG_LEVEL", LOG_LEVEL_DEFAULT),
    )
