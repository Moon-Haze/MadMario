import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ClipReward, RepeatAction, ResizeObservation


class StuckHandler(gym.Wrapper):
    """检测"水平方向长时间无净推进"，先随机接管动作尝试脱困，再兜底判负。

    判定口径：维护一个停滞窗口，窗口内只增不减的最大 x 相对窗口起点推进
    不足 ``progress_epsilon`` 时累加 ``stuck_steps``。只看 ``x_pos``，不看
    ``y_pos``——原地反复跳跃而 x 不前进同样属于卡住。

    状态机：

    - NORMAL：停滞满 ``max_stuck_steps`` 且 ``rescue_enabled`` 时进入 RESCUE；
      未开启脱困时按 ``penalty_enabled`` 直接判负（旧行为）。
    - RESCUE：忽略传入动作，改用 ``action_space.sample()`` 随机尝试；取得净
      推进即回到 NORMAL。
    - 救援满 ``rescue_max_steps`` 仍无进展时：``penalty_enabled`` 为真则扣分
      并截断，否则原地开始新一轮救援（播放模式：永不因卡住而结束关卡）。
    """

    def __init__(self, env, max_stuck_steps, penalty, progress_epsilon=8.0,
                 penalty_enabled=True, rescue_enabled=True, rescue_max_steps=120):
        super().__init__(env)
        self.max_stuck_steps = int(max_stuck_steps)
        self.penalty = float(penalty)
        self.progress_epsilon = float(progress_epsilon)
        self.penalty_enabled = bool(penalty_enabled)
        self.rescue_enabled = bool(rescue_enabled)
        self.rescue_max_steps = int(rescue_max_steps)
        self.best_x = 0.0
        self.window_start_x = 0.0
        self.window_best_x = 0.0
        self.stuck_steps = 0
        self.rescue_steps = 0
        self.rescuing = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        x_pos = self._position_value(info.get("x_pos"))
        self.best_x = x_pos if x_pos is not None else 0.0
        self.window_start_x = self.best_x
        self.window_best_x = self.best_x
        self.stuck_steps = 0
        self.rescue_steps = 0
        self.rescuing = False
        info["max_x_pos"] = self.best_x
        info["stuck_steps"] = self.stuck_steps
        info["rescuing"] = self.rescuing
        info["rescue_steps"] = self.rescue_steps
        return obs, info

    def step(self, action):
        if self.rescuing:
            # 脱困阶段接管动作：确定性策略会被同一障碍反复挡住，只有随机采样才能打破死循环
            action = self.action_space.sample()

        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        x_pos = self._position_value(info.get("x_pos"))

        if x_pos is not None:
            self.best_x = max(self.best_x, x_pos)
            self.window_best_x = max(self.window_best_x, x_pos)
            if self.window_best_x - self.window_start_x >= self.progress_epsilon:
                # 取得进展：窗口基准前移、停滞计数清零、结束脱困
                self.window_start_x = self.window_best_x
                self.stuck_steps = 0
                self.rescuing = False
                self.rescue_steps = 0
            elif self.rescuing:
                self.rescue_steps += 1
            else:
                self.stuck_steps += 1

        if not terminated and not truncated and not info.get("flag_get"):
            if not self.rescuing and self.stuck_steps >= self.max_stuck_steps:
                if self.rescue_enabled:
                    # 当步判定之后置位，因此下一步的动作才被接管
                    self.rescuing = True
                    self.rescue_steps = 0
                elif self.penalty_enabled:
                    reward = float(reward) - self.penalty
                    truncated = True
                    info["stuck"] = True
            elif self.rescuing and self.rescue_steps >= self.rescue_max_steps:
                if self.penalty_enabled:
                    self.rescuing = False
                    reward = float(reward) - self.penalty
                    truncated = True
                    info["stuck"] = True
                else:
                    # 播放语义：不判负，原地开始新一轮随机尝试（不留策略动作的空隙）
                    self.rescue_steps = 0

        info["max_x_pos"] = self.best_x
        info["stuck_steps"] = self.stuck_steps
        info["rescuing"] = self.rescuing
        info["rescue_steps"] = self.rescue_steps
        return obs, reward, terminated, truncated, info

    def _position_value(self, value):
        if value is None:
            return None
        return float(value)


class ProgressReward(gym.Wrapper):
    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = float(scale)
        self.last_x_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_x_pos = info.get("x_pos")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_pos = info.get("x_pos")
        if x_pos is not None and self.last_x_pos is not None:
            delta = float(x_pos) - self.last_x_pos
            reward = float(reward) + self.scale * max(0.0, delta)
        if x_pos is not None:
            self.last_x_pos = float(x_pos)
        return obs, reward, terminated, truncated, info


class ScoreReward(gym.Wrapper):
    """根据游戏得分（score）变化给予额外奖励。

    追踪 ``info["score"]`` 的变化量，将得分增加转化为额外奖励。
    放在 ``ClipReward`` 之后使用，保证得分变化不被裁剪。
    """

    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = float(scale)
        self.last_score = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_score = info.get("score")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        score = info.get("score")
        if score is not None and self.last_score is not None:
            delta = float(score) - self.last_score
            if delta > 0:
                reward = float(reward) + self.scale * delta
        if score is not None:
            self.last_score = float(score)
        return obs, reward, terminated, truncated, info


class FlagReward(gym.Wrapper):
    """拉旗子通关时给予一次性大额奖励。

    检测 ``info["flag_get"]``，为 True 时添加固定奖励。
    放在所有裁剪类包装器之后，确保奖励值完整保留。
    """

    def __init__(self, env, reward_value=50.0):
        super().__init__(env)
        self.reward_value = float(reward_value)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("flag_get"):
            reward = float(reward) + self.reward_value
        return obs, reward, terminated, truncated, info


