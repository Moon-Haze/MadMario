import copy

from torch import nn


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, _, _ = input_dim
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.value = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, input):
        features = self.features(input)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class MarioNet(nn.Module):
    """Dueling CNN 结构：卷积特征 -> value/advantage 双流 -> Q 值。"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        _, h, w = input_dim

        if h != 84:
            raise ValueError(f"期望输入高度为: 84, 实际为: {h}")
        if w != 84:
            raise ValueError(f"期望输入宽度为: 84, 实际为: {w}")

        self.online = DuelingDQN(input_dim, output_dim)
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        if model == "target":
            return self.target(input)
        raise ValueError(f"未知模型类型: {model}")
