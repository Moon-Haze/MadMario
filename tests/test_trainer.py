"""测试 trainer.py 的配置换算与续训接线。

train() 本身会起 SubprocVecEnv 真跑环境，这里只测从它里面抽出来的纯函数：
它们是三处修复各自的决策点，且都能脱离 SB3 运行时验证。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from stable_baselines3.common.utils import LinearSchedule

from mad_mario.config import AppConfig, EnvConfig, SB3ArtifactConfig, SB3Config
from mad_mario.training.trainer import (
    _apply_exploration_schedule,
    _assign_levels,
    _make_eval_thunks,
    _resume_hyperparams,
    _resolve_exploration_fraction,
    _select_eval_levels,
)


def _app_config(*, levels=None, eval_levels=3, world=1, level=1, seed=42, num_envs=4) -> AppConfig:
    return AppConfig(
        env=EnvConfig(world=world, level=level, levels=levels),
        sb3=SB3Config(seed=seed),
        artifacts=SB3ArtifactConfig(eval_levels=eval_levels, num_envs=num_envs),
    )


class TestAssignLevels:
    """多关卡分配必须吃 seed：--seed 决定每个子环境分到哪一关。

    此前分配走全局 random，seed 只进 SB3 构造器管不到这里，同 seed
    两次运行分出不同的环境组合——可复现性在第一步就断了。
    """

    LEVELS = ["1-1", "1-2", "1-3", "1-4"]

    def test_same_seed_same_assignment(self):
        cfg = _app_config(levels=self.LEVELS)
        assert _assign_levels(cfg) == _assign_levels(cfg)

    def test_different_seed_reshuffles(self):
        """两个 seed 各抽 12 次：序列全同的概率是 (1/4)^12，可忽略。"""
        a = _assign_levels(_app_config(levels=self.LEVELS, seed=1, num_envs=12))
        b = _assign_levels(_app_config(levels=self.LEVELS, seed=2, num_envs=12))
        assert a != b

    def test_draws_from_levels_only(self):
        assigned = _assign_levels(_app_config(levels=self.LEVELS, num_envs=16))
        assert len(assigned) == 16
        assert set(assigned) <= set(self.LEVELS)

    def test_single_level_returns_none_per_env(self):
        """单关卡（或未启用 levels）时全 None，_env_thunk 改读 world/level。"""
        assert _assign_levels(_app_config(num_envs=4)) == [None] * 4
        assert _assign_levels(_app_config(levels=["1-1"], num_envs=4)) == [None] * 4


class TestResolveExplorationFraction:
    """ε 退火窗口的换算：比例语义会随总步数放大，绝对步数不会。"""

    def test_plain_fraction_passes_through(self):
        sb3 = SB3Config(exploration_fraction=0.3)
        assert _resolve_exploration_fraction(sb3, num_timesteps=0) == 0.3

    def test_absolute_steps_are_relative_to_new_run_total(self):
        sb3 = SB3Config(total_timesteps=1_000_000, exploration_steps=100_000)
        assert _resolve_exploration_fraction(sb3, num_timesteps=0) == pytest.approx(0.1)

    def test_total_includes_already_trained_steps(self):
        """分母必须是 total + 已训步数，才是 SB3 的 _total_timesteps。"""
        sb3 = SB3Config(total_timesteps=1_000, exploration_steps=100)
        assert _resolve_exploration_fraction(sb3, num_timesteps=9_000) == pytest.approx(0.01)

    def test_target_already_passed_collapses_to_final_eps(self):
        """目标步数落在身后时 ε 立刻跌到 final——这是绝对步数该有的语义。"""
        sb3 = SB3Config(total_timesteps=1_000, exploration_steps=10)
        num_timesteps = 5_000
        fraction = _resolve_exploration_fraction(sb3, num_timesteps)

        schedule = LinearSchedule(1.0, 0.02, fraction)
        progress_remaining = 1 - num_timesteps / (sb3.total_timesteps + num_timesteps)
        assert schedule(progress_remaining) == 0.02

    def test_zero_total_does_not_divide_by_zero(self):
        """总步数为 0 是畸形输入，退回比例而不是抛 ZeroDivisionError。"""
        sb3 = SB3Config(total_timesteps=0, exploration_steps=100, exploration_fraction=0.4)
        assert _resolve_exploration_fraction(sb3, num_timesteps=0) == 0.4


class TestApplyExplorationSchedule:
    """退火表落地：新训练从初始 ε 起退，续训按已有进度接着算。"""

    def _fake_model(self):
        """只用到这三个属性；SB3 的 DQN 在 __init__ 里把 rate 占位成 0.0。"""
        return SimpleNamespace(
            exploration_fraction=0.1, exploration_schedule=None, exploration_rate=0.0
        )

    def test_new_run_starts_at_initial_eps(self):
        model = self._fake_model()
        sb3 = SB3Config(total_timesteps=20_000, exploration_steps=10_000)
        _apply_exploration_schedule(model, sb3, num_timesteps=0)

        assert model.exploration_fraction == pytest.approx(0.5)
        assert model.exploration_rate == pytest.approx(1.0)

    def test_resumed_run_resumes_the_archived_progress(self):
        """续训时 ε 必须按已有步数接着算，而不是从头再退一遍。"""
        model = self._fake_model()
        sb3 = SB3Config(total_timesteps=1_000_000, exploration_fraction=0.1)
        _apply_exploration_schedule(model, sb3, num_timesteps=90_000)

        # 已走到总量的 8.26%，落在 10% 的退火窗口内 → ε 略高于 final_eps
        assert 0.02 < model.exploration_rate < 0.2


class TestResumeHyperparams:
    """续训时哪些超参以 CLI 为准。"""

    def _dqn_params(self):
        return {
            "policy": "CnnPolicy",
            "learning_rate": 1e-4,
            "buffer_size": 200_000,
            "batch_size": 64,
            "n_steps": 3,
            "gamma": 0.99,
            "policy_kwargs": {},
            "env": object(),
            "tensorboard_log": "x",
        }

    def test_carries_cli_hyperparams(self):
        picked = _resume_hyperparams(self._dqn_params())
        assert picked["buffer_size"] == 200_000
        assert picked["batch_size"] == 64
        assert picked["n_steps"] == 3

    def test_drops_structural_params(self):
        """policy / env / policy_kwargs 不能覆盖：权重会与结构对不上号。"""
        picked = _resume_hyperparams(self._dqn_params())
        assert "policy" not in picked
        assert "env" not in picked
        assert "policy_kwargs" not in picked
        assert "tensorboard_log" not in picked

    def test_excludes_exploration_fraction(self):
        """退火比例由 _apply_exploration_schedule 统一处理，不走这条路。"""
        assert "exploration_fraction" not in _resume_hyperparams(self._dqn_params())


class TestSelectEvalLevels:
    """评估关卡抽样：必须落在训练分布上，且跨次可比。"""

    def test_single_level_uses_configured_world_level(self):
        assert _select_eval_levels(_app_config(levels=None, world=2, level=3)) == [
            "SuperMarioBros-2-3-v0"
        ]

    def test_single_level_list_uses_that_level(self):
        config = _app_config(levels=["SuperMarioBros-4-1-v0"])
        assert _select_eval_levels(config) == ["SuperMarioBros-4-1-v0"]

    def test_picks_evenly_spaced_subset(self):
        levels = [f"SuperMarioBros-1-{i}-v0" for i in range(1, 5)] + [
            f"SuperMarioBros-2-{i}-v0" for i in range(1, 5)
        ]
        config = _app_config(levels=levels, eval_levels=2)
        assert _select_eval_levels(config) == [levels[0], levels[4]]

    def test_all_levels_when_request_covers_everything(self):
        levels = ["SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0"]
        config = _app_config(levels=levels, eval_levels=8)
        assert _select_eval_levels(config) == levels

    def test_is_deterministic(self):
        levels = [f"SuperMarioBros-1-{i}-v0" for i in range(1, 5)]
        config = _app_config(levels=levels, eval_levels=2)
        assert _select_eval_levels(config) == _select_eval_levels(config)

    def test_thunks_cover_the_selected_levels(self):
        levels = ["SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0"]
        assert len(_make_eval_thunks(_app_config(levels=levels, eval_levels=2))) == 2
