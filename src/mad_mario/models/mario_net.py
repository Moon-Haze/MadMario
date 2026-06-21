import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.epsilon_weight = epsilon_out.outer(epsilon_in)
        self.epsilon_bias = epsilon_out.clone()

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.epsilon_weight
            bias = self.bias_mu + self.bias_sigma * self.epsilon_bias
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


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


class NoisyDuelingDQN(DuelingDQN):
    def __init__(self, input_dim, output_dim, std_init=0.5):
        super(DuelingDQN, self).__init__()
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
            NoisyLinear(3136, 512, std_init=std_init),
            nn.ReLU(),
            NoisyLinear(512, 1, std_init=std_init),
        )
        self.advantage = nn.Sequential(
            NoisyLinear(3136, 512, std_init=std_init),
            nn.ReLU(),
            NoisyLinear(512, output_dim, std_init=std_init),
        )

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class MarioNet(nn.Module):
    """Dueling CNN 结构：卷积特征 -> value/advantage 双流 -> Q 值。

    当 noisy_std_init > 0 时使用 NoisyDuelingDQN（替代 epsilon-greedy 探索）。
    """

    def __init__(self, input_dim, output_dim, noisy_std_init=0.5):
        super().__init__()
        _, h, w = input_dim

        if h != 84:
            raise ValueError(f"期望输入高度为: 84, 实际为: {h}")
        if w != 84:
            raise ValueError(f"期望输入宽度为: 84, 实际为: {w}")

        self._noisy = noisy_std_init > 0
        if self._noisy:
            self.online = NoisyDuelingDQN(input_dim, output_dim, std_init=noisy_std_init)
        else:
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

    def reset_noise(self):
        if self._noisy:
            self.online.reset_noise()
