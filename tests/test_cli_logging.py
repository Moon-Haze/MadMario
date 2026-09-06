"""测试 cli.py 的日志装配接线（mock 掉重依赖，验证调用关系）。"""

from __future__ import annotations

import logging
from types import SimpleNamespace


def fake_setup_logging_factory(log_path=None):
    def _make(calls):
        def fake_setup_logging(level=logging.INFO, log_dir=None, prefix="train"):
            calls["level"] = level
            calls["log_dir"] = log_dir
            calls["prefix"] = prefix
            return log_path
        return fake_setup_logging
    return _make


class TestMainTrainWiring:
    def test_configures_file_logging_and_passes_log_path(self, tmp_path, monkeypatch):
        import mad_mario.cli as cli_mod

        fake_save_root = tmp_path / "saves"
        fake_config = SimpleNamespace(artifacts=SimpleNamespace(save_root=fake_save_root))
        fake_log_path = tmp_path / "logs" / "train_20260906_150422.log"

        monkeypatch.setattr(cli_mod, "config_from_train_args", lambda args: fake_config)
        calls: dict = {}
        monkeypatch.setattr(cli_mod, "setup_logging", fake_setup_logging_factory(fake_log_path)(calls))
        train_calls: dict = {}

        def fake_train(config, log_path=None):
            train_calls["config"] = config
            train_calls["log_path"] = log_path

        monkeypatch.setattr(cli_mod, "train", fake_train)
        monkeypatch.setattr("sys.argv", ["mad-mario", "train", "--log-level", "debug"])

        cli_mod.main()

        assert calls["log_dir"] == fake_save_root / "logs"
        assert calls["level"] == logging.DEBUG
        assert train_calls["config"] is fake_config
        assert train_calls["log_path"] is fake_log_path  # setup_logging 返回值被透传给 train


class TestMainPlayWiring:
    def test_terminal_only_and_reaches_play(self, tmp_path, monkeypatch):
        import mad_mario.cli as cli_mod

        fake_config = SimpleNamespace()
        monkeypatch.setattr(cli_mod, "config_from_play_args", lambda args: fake_config)
        calls: dict = {}
        monkeypatch.setattr(cli_mod, "setup_logging", fake_setup_logging_factory()(calls))
        play_calls: dict = {}

        def fake_play(config):
            play_calls["config"] = config

        monkeypatch.setattr(cli_mod, "play", fake_play)
        monkeypatch.setattr("sys.argv", ["mad-mario", "play"])

        cli_mod.main()

        assert calls["log_dir"] is None
        assert calls["level"] == logging.INFO
        assert play_calls["config"] is fake_config
