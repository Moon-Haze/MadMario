from collections import deque

import numpy as np
from tqdm import tqdm


def rolling_mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


class TrainingProgress:
    def __init__(self, total, initial=0, desc="训练进度"):
        self.progress_bar = tqdm(total=total, initial=initial, desc=desc, unit="回合", position=0)
        self.status_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=False)

    def update(self, count=1):
        self.progress_bar.update(count)

    def set_single_status(self, agent, ep_reward, ep_length, last_loss, last_q):
        self.status_bar.set_description_str(
            f"步数={agent.curr_step} | "
            f"探索率={agent.exploration_rate:.3f} | "
            f"回合奖励={ep_reward:.1f} | "
            f"回合长度={ep_length} | "
            f"损失={'-' if last_loss is None else f'{last_loss:.4f}'} | "
            f"Q值={'-' if last_q is None else f'{last_q:.4f}'}"
        )

    def set_vector_status(self, config, agent, recent_rewards, last_loss, last_q):
        self.status_bar.set_description_str(
            f"环境数={config.num_envs} | "
            f"步数={agent.curr_step} | "
            f"探索率={agent.exploration_rate:.3f} | "
            f"近100回合奖励={rolling_mean(recent_rewards):.1f} | "
            f"损失={'-' if last_loss is None else f'{last_loss:.4f}'} | "
            f"Q值={'-' if last_q is None else f'{last_q:.4f}'}"
        )

    def close(self):
        self.status_bar.close()
        self.progress_bar.close()


def recent_metric_buffers():
    return deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
