import argparse
import os
import sys
import pandas as pd

# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from envs import SumoGridEnvironment
from sumo_rl.agents.ql_agent import QLAgent
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
import numpy as np
import torch
from torch import nn


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


class ActorCritic(TorchModelV2, nn.Module):
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
        print(f"times called: {self.times_called}")
        return model_out, state

    def value_function(self):
        return self._global_value_out

# Register the model and the environment
ModelCatalog.register_custom_model("grid_actor_critic", ActorCritic)
register_env(
    "3x3grid",
    lambda _: SumoGridEnvironment(
        net_file='nets/3x3/3x3.net.xml',
        route_folder='nets/3x3/',
        num_cols=3,
        num_rows=3,
        num_runs=30,
        use_gui=False,
        num_seconds=1000,
        time_to_load_vehicles=2,
        max_depart_delay=0,
        out_csv_name="outputs/3x3grid/ppo",
    ),
)

ray.init()

trainer = PPOTrainer(
    env="3x3grid",
    config={
        "framework": "torch",
        "train_batch_size": 10000,
        "num_workers": 4,
        "sgd_minibatch_size": 16,
        "num_gpus": 0.5,
        "model": {
            "custom_model": "grid_actor_critic",
            "custom_model_config": {
                "num_tls": 3 * 3,
            },
        },
    }
)

result = trainer.train()
print(result)


# if __name__ == '__main__':
#     alpha = 0.1
#     gamma = 0.99
#     decay = 1
#     runs = 1

#     env = SumoGridEnvironment(
#         net_file='nets/3x3/3x3.net.xml',
#         route_file='nets/3x3/3x3-peak_3.rou.xml',
#         num_cols=3,
#         num_rows=3,
#         use_gui=False,
#         num_seconds=14500,
#         time_to_load_vehicles=2,
#         max_depart_delay=0,
#         phases=[
#             traci.trafficlight.Phase(42, "GGGrrrrrGGGrrrrr"),
#             traci.trafficlight.Phase(3, "GyyrrrrrGyyrrrrr"),
#             traci.trafficlight.Phase(42, "GrrGrrrrGrrGrrrr"),
#             traci.trafficlight.Phase(3, "yrryrrrryrryrrrr"),
#             traci.trafficlight.Phase(42, "rrrrGGGrrrrrGGGr"),
#             traci.trafficlight.Phase(3, "rrrrGyyrrrrrGyyr"),
#             traci.trafficlight.Phase(42, "rrrrGrrGrrrrGrrG"),
#             traci.trafficlight.Phase(3, "rrrryrryrrrryrry"),
#         ],
#     )

#     for run in range(1, runs + 1):
#         initial_states = env.reset()
#         initial_states = np.array_split(initial_states, 4, axis=1)
#         tmp = []
#         for init_state in initial_states:
#             tmp.extend(np.array_split(initial_states, 4, axis=2))
#         ql_agents = {
#             ts: QLAgent(
#                 starting_state=initial_states[ts_idx],
#                 state_space=env.observation_space,
#                 action_space=env.action_space,
#                 alpha=alpha,
#                 gamma=gamma,
#                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay)
#             ) for ts_idx, ts in enumerate(env.ts_ids)
#         }
#         done = {'__all__': False}
#         while not done['__all__']:
#             actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

#             s, r, done, _ = env.step(action=actions)

#             for agent_id in ql_agents.keys():
#                 ql_agents[agent_id].learn(next_state=env.encode(s[agent_id]), reward=r[agent_id])

#         infos = env.metrics
#         env.close()

#         df = pd.DataFrame(infos)
#         df.to_csv('outputs/3x3grid/c2_alpha{}_gamma{}_decay{}_run{}.csv'.format(alpha, gamma, decay, run), index=False)
