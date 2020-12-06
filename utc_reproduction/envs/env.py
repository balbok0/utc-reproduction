# Adapted from: https://github.com/LucasAlegre/sumo-rl

import os
import sys
import traci
import sumolib
from gym import Env
import traci.constants as tc
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, MultiAgentDict
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import DefaultDict, Dict, List

from .traffic_signal import TrafficSignal
from .network import SumoGridNetwork

class SumoGridEnvironment(MultiAgentEnv):
    def __init__(
        self,
        net_file: str,
        route_folder: str,
        num_cols: int,
        num_rows: int,
        num_train_steps: int,
        use_gui: bool = False,
        num_seconds: int = 20000,
        max_depart_delay: int = 100000,
        time_to_teleport: int = -1,
        time_to_load_vehicles: int = 0,
        delta_time: int = 5,
        out_csv_name: str = None,
    ):
        self._net = net_file
        self._route_dir = route_folder
        self._agent_routes = dict((f"agent_{x}", y) for x, y in enumerate(list(Path(self._route_dir).rglob("*.rou.xml"))[:1]))

        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self.num_cols = num_cols
        self.num_rows = num_rows

        self.run = 0
        self.step_num = 0
        self.num_train_steps = num_train_steps

        # (num observables, 4 * rows, 4 * cols). Assumes at most 4 directions, 2 incoming lanes each
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones((2, 4 * self.num_rows, 4 * self.num_cols)),
            high=np.inf * np.ones((2, 4 * self.num_rows, 4 * self.num_cols)),
        )
        # (rows, )
        self.action_space = spaces.MultiDiscrete([2] * 9)

        self.agent_sumo_envs: Dict[str, SumoGridNetwork] = {}
        self.traffic_signals: DefaultDict[str, Dict[str, TrafficSignal]] = defaultdict(dict)

        self.sim_max_time = num_seconds
        self.time_to_load_vehicles = time_to_load_vehicles  # number of simulation seconds ran in reset() before learning starts
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport

        self.metrics: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self.out_csv_name = out_csv_name

    @property
    def sim_step(self):
        return traci.simulation.getTime()

    def reset(self):
        if self.run != 0:
            traci.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.step_num = 0

        # Initialize SUMO environments for agents
        for agent_id, route_file in self._agent_routes.items():
            sumo_cmd = [
                self._sumo_binary,
                '-n', self._net,
                '-r', route_file,
                '--max-depart-delay', str(self.max_depart_delay),
                '--waiting-time-memory', '10000',
                '--time-to-teleport', str(self.time_to_teleport),
                '--random'
            ]
            if self.use_gui:
                sumo_cmd.append('--start')

            traci.start(sumo_cmd, label=agent_id)

            # Build networks for each environment
            self.agent_sumo_envs[agent_id] = SumoGridNetwork(agent_id, self.num_rows, self.num_cols)
            self.agent_sumo_envs[agent_id].reset()

        return self._compute_observations()

    def step(self, action_dict: MultiAgentDict):
        self.step_num += 1
        print(f"Current step number: {self.step_num}")
        if action_dict is None:
            for _, net in self.agent_sumo_envs.items():
                net.step(self.delta_time)
        else:
            for agent_id, actions in action_dict.items():
                net = self.agent_sumo_envs[agent_id]
                net.apply_actions(actions)

                net.step(self.delta_time)

        observations = self._compute_observations()
        rewards = {}
        infos: Dict[str, Dict[str, float]] = {}
        for agent_id, net in self.agent_sumo_envs.items():
            rewards[agent_id] = net.reward(beta=min(1.0, max(self.step_num * 1. / self.num_train_steps, 0.0)))
            infos[agent_id] = net.info()
            infos[agent_id]["reward"] = rewards[agent_id]
            infos[agent_id]["step_time"] = self.sim_step
            self.metrics[agent_id].append(infos[agent_id])
        dones = {'__all__': self.sim_step > self.sim_max_time}

        return observations, rewards, dones, infos

    def _compute_observations(self):
        return {agent: network.as_feature_grid() for agent, network in self.agent_sumo_envs.items()}

    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            for agent_id, agent_infos in self.metrics.items():
                df = pd.DataFrame(self.metrics)

                df.to_csv(
                    f"out_csv_name_agent_id_{agent_id}_run_{run}.csv",
                    index=False,
                )
