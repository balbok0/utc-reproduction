import argparse
import os
import sys
import pandas as pd

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from envs import SumoEnvironment
from sumo_rl.agents.ql_agent import QLAgent
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
import numpy as np


if __name__ == '__main__':
    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 1

    env = SumoEnvironment(
        net_file='nets/3x3/3x3.net.xml',
        route_file='nets/3x3/3x3-peak_3.rou.xml',
        use_gui=False,
        num_seconds=14500,
        time_to_load_vehicles=300,
        max_depart_delay=0,
        phases=[
            traci.trafficlight.Phase(42, "GGGrrrrrGGGrrrrr"),
            traci.trafficlight.Phase(3, "GyyrrrrrGyyrrrrr"),
            traci.trafficlight.Phase(42, "GrrGrrrrGrrGrrrr"),
            traci.trafficlight.Phase(3, "yrryrrrryrryrrrr"),
            traci.trafficlight.Phase(42, "rrrrGGGrrrrrGGGr"),
            traci.trafficlight.Phase(3, "rrrrGyyrrrrrGyyr"),
            traci.trafficlight.Phase(42, "rrrrGrrGrrrrGrrG"),
            traci.trafficlight.Phase(3, "rrrryrryrrrryrry"),
        ],
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()
        ql_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states[ts]),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay)
            ) for ts in env.ts_ids
        }
        done = {'__all__': False}
        while not done['__all__']:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, _ = env.step(action=actions)

            for agent_id in ql_agents.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id]), reward=r[agent_id])

        infos = env.metrics
        env.close()

        df = pd.DataFrame(infos)
        df.to_csv('outputs/3x3grid/c2_alpha{}_gamma{}_decay{}_run{}.csv'.format(alpha, gamma, decay, run), index=False)
