"""CLI 入口单元测试（不含真实网络调用）。"""

from __future__ import annotations

import pytest

from rag_agent.cli import main
from rag_agent.config import load_settings


def test_missing_command_exits_with_usage(capsys):
    """不传子命令时按 argparse 惯例打印用法并以退出码 2 结束。"""
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_ask_without_milvus_config_fails_cleanly(no_env, monkeypatch):
    """缺 Milvus 配置时 ask 应抛出带明确提示的 RuntimeError，而不是晦涩的栈追踪。"""
    # 屏蔽 main() 内部对项目 .env 的自动发现，保证测试不触碰真实服务
    monkeypatch.setattr("rag_agent.cli.load_settings", lambda: load_settings(no_env))
    with pytest.raises(RuntimeError, match="MILVUS_URI / MILVUS_TOKEN"):
        main(["ask", "知识库什么时候更新？"])


def test_ingest_missing_dir_fails(no_env, monkeypatch, tmp_path):
    """ingest 指向不存在的目录时抛 FileNotFoundError。"""
    monkeypatch.setattr("rag_agent.cli.load_settings", lambda: load_settings(no_env))
    with pytest.raises(FileNotFoundError):
        main(["ingest", str(tmp_path / "not_exist")])
