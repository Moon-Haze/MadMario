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
        # cli.main() 里是延迟导入（train 不是 cli 的模块属性）。补丁要打在
        # 来源模块上——main() 执行到那行才去查这个属性
        import mad_mario.training.trainer as trainer_mod

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

        monkeypatch.setattr(trainer_mod, "train", fake_train)
        monkeypatch.setattr("sys.argv", ["mad-mario", "train", "--log-level", "debug"])

        cli_mod.main()

        assert calls["log_dir"] == fake_save_root / "logs"
        assert calls["level"] == logging.DEBUG
        assert train_calls["config"] is fake_config
        assert train_calls["log_path"] is fake_log_path  # setup_logging 返回值被透传给 train


class TestHeadlessBackend:
    def test_importing_metrics_pins_agg(self, monkeypatch):
        """训练路径的 Agg 契约由 metrics 模块自己保证——导入它就该锁后端。

        TkAgg 下主线程建出的 Tk 对象只能由创建线程销毁，训练中录制最优
        模型的后台线程一旦经 GC 回收它们，进程会被 Tcl 终止。后端必须在
        pyplot 首次导入之前定死：spy 顶替 pyplot，metrics 若在 use 之前
        碰它（哪怕只是设置 rcParams），这里就会当场红。
        """
        import importlib
        import sys

        import matplotlib

        chosen: list[str] = []

        class _SpyPyplot:
            """只放行 rcParams（模块级合法配置）；use 之前的任何其他
            pyplot 访问都说明顺序反了。"""

            def __getattr__(self, name):
                if name == "rcParams":
                    return {}
                if name.startswith("_"):
                    # __spec__ 等导入机制属性，与后端顺序无关
                    raise AttributeError(name)
                raise AssertionError(
                    f"metrics 在 matplotlib.use('Agg') 之前碰了 pyplot.{name}"
                )

        real_pyplot = sys.modules.get("matplotlib.pyplot")
        monkeypatch.setattr(matplotlib, "use", chosen.append)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", _SpyPyplot())
        try:
            import mad_mario.logging.metrics as metrics_mod

            importlib.reload(metrics_mod)
        finally:
            # 把真 pyplot 放回去并再加载一次，让模块带着真绑定继续服务
            # 本会话后面的测试（test_metrics 要真画 PNG）
            if real_pyplot is not None:
                monkeypatch.setitem(sys.modules, "matplotlib.pyplot", real_pyplot)
            else:
                monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)
            import mad_mario.logging.metrics as metrics_mod

            importlib.reload(metrics_mod)

        assert chosen and chosen[0] == "Agg", "导入 metrics 的第一件事必须是锁 Agg"


class TestMainPlayWiring:
    def test_terminal_only_and_reaches_play(self, tmp_path, monkeypatch):
        import mad_mario.cli as cli_mod
        # 同上：play 也是延迟导入，补丁打在来源模块上
        import mad_mario.playback.runner as runner_mod

        fake_config = SimpleNamespace()
        monkeypatch.setattr(cli_mod, "config_from_play_args", lambda args: fake_config)
        calls: dict = {}
        monkeypatch.setattr(cli_mod, "setup_logging", fake_setup_logging_factory()(calls))
        play_calls: dict = {}

        def fake_play(config):
            play_calls["config"] = config

        monkeypatch.setattr(runner_mod, "play", fake_play)
        monkeypatch.setattr("sys.argv", ["mad-mario", "play"])

        cli_mod.main()

        assert calls["log_dir"] is None
        assert calls["level"] == logging.INFO
        assert play_calls["config"] is fake_config
