import numpy as np
import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class Block(nn.Module):
    # from: https://stackoverflow.com/questions/60817390/implementing-a-simple-resnet-block-with-pytorch
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(Block, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = self.relu(out)

        return out

class ActorTorch(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=4)
        self.linear_1 = nn.Linear(256, 32)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(32, 2)

    def forward(self, x):
        # Input: <batch, 256, n_tls * 4, n_tls * 4>
        x = self.conv_1(x)
        # <batch, 512, n_tls, n_tls>
        x = x.view((len(x), 256, -1))
        x = torch.transpose(x, 1, 2)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = x.view((len(x), -1))
        return x

class CriticTorch(nn.Module):
    def __init__(self, num_tls):
        super(self.__class__, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=4)
        self.linear_1 = nn.Linear(256, 32)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(32, 1)

        self.linear_global = nn.Linear(num_tls, 1)

    def forward(self, x):
        # Input: <batch, 256, n_tls * 4, n_tls * 4>
        x = self.conv_1(x)
        # <batch, 512, n_tls, n_tls>
        x = x.view((len(x), 256, -1))
        x = torch.transpose(x, 1, 2)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = torch.squeeze(x)

        g = self.linear_global(x)
        return x, g


class ActorCriticTorch(nn.Module):
    def __init__(self, num_tls):
        super(self.__class__, self).__init__()

        self.init_sequence = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            Block(8, 32),
            Block(32, 64),
            Block(64, 128),
            Block(128, 256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.actor_sequence = ActorTorch()
        self.critic_sequence = CriticTorch(num_tls)

    def forward(self, x):
        x = self.init_sequence(x)
        return self.actor_sequence(x), self.critic_sequence(x)


class Grid3x3Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, num_tls):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.base_model = ActorCriticTorch(num_tls)
        self.times_called = 0

    def forward(self, input_dict, state, seq_lens):
        model_out, (self._local_value_out, self._global_value_out) = self.base_model(
            input_dict["obs"],
        )
        self.times_called += 1
        # print(f"times called: {self.times_called}")
        return model_out, state

    def value_function(self):
        return self._global_value_out
