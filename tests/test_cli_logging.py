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
        # cli.main() 里是延迟导入（要抢在 pyplot 之前定 matplotlib 后端，见
        # cli._use_headless_backend），train 不是 cli 的模块属性。补丁要打在
        # 来源模块上——main() 执行到那行才去查这个属性
        import mad_mario.training.trainer as trainer_mod

        fake_save_root = tmp_path / "saves"
        fake_config = SimpleNamespace(artifacts=SimpleNamespace(save_root=fake_save_root))
        fake_log_path = tmp_path / "logs" / "train_20260906_150422.log"

        monkeypatch.setattr(cli_mod, "config_from_train_args", lambda args: fake_config)
        # 后端切换是全局状态，这里让它空转：本测试只关心日志接线，而真跑一遍会
        # 把整个 pytest 会话的 matplotlib 后端钉死在 Agg 上，后面的 test_viewer
        # 就用不了交互后端了。后端本身由 TestHeadlessBackend 单独测
        monkeypatch.setattr(cli_mod, "_use_headless_backend", lambda: None)
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
    def test_pins_agg(self, monkeypatch):
        """训练路径必须挑 Agg——TkAgg 下有被后台录制线程回收 Tk 对象的风险。

        用打桩的 matplotlib.use 断言，不真改全局后端：真改会污染同一会话里
        其它依赖交互后端的测试。
        """
        import matplotlib

        import mad_mario.cli as cli_mod

        chosen: list[str] = []
        monkeypatch.setattr(matplotlib, "use", chosen.append)

        cli_mod._use_headless_backend()

        assert chosen == ["Agg"], "必须是 Agg（只写文件、不开窗口）"


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
