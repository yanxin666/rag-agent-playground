"""基于 stdlib logging 的统一日志初始化。"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """初始化全局日志：统一格式并输出到 stderr。

    level 为日志级别字符串（DEBUG/INFO/WARNING/...），非法值回退 INFO。
    force=True 保证重复调用也能覆盖已生效的旧配置（如测试中多次初始化）。
    """
    logging.basicConfig(
        # "INFO" -> logging.INFO；取不到对应属性时用 INFO 兜底
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        # 日志走 stderr、print 结果走 stdout，两者互不干扰
        stream=sys.stderr,
        force=True,
    )
