import argparse
import os
import sys
import pandas as pd
import traci
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from utc_reproduction import SumoGridEnvironment, Grid3x3Model


# Register the model and the environment
ModelCatalog.register_custom_model("grid_actor_critic", Grid3x3Model)
register_env(
    "3x3grid",
    lambda _: SumoGridEnvironment(
        net_file='nets/3x3/3x3.net.xml',
        route_folder='nets/3x3/',
        num_cols=3,
        num_rows=3,
        num_runs=30,
        use_gui=False,
        num_seconds=14050,
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
