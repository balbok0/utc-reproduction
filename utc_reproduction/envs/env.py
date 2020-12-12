# Adapted from: https://github.com/LucasAlegre/sumo-rl

import os
import traci
import sumolib
from gym import Env
from gym import spaces
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import DefaultDict, Dict, List

from .traffic_signal import TrafficSignal
from .network import SumoGridNetwork


class SumoGridEnvironment(Env):
    def __init__(
        self,
        net_file: str,
        route_folder: str,
        num_cols: int,
        num_rows: int,
        num_train_steps: int,
        additionals_file: str = None,
        use_gui: bool = False,
        num_seconds: int = 20000,
        max_depart_delay: int = 100000,
        time_to_teleport: int = -1,
        time_to_load_vehicles: int = 2,
        delta_time: int = 5,
        out_csv_name: str = None,
    ):
        self._net = net_file
        self._additionals_file = additionals_file
        self._route_dir = route_folder
        self._route_files = list(Path(self._route_dir).rglob("*.rou.xml"))
        self.agent_id = f"agent_{np.random.randint(0, 999999999)}"

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

        self.traffic_signals: DefaultDict[str, Dict[str, TrafficSignal]] = defaultdict(dict)

        self.sim_max_time = num_seconds
        # number of simulation seconds ran in reset() before learning starts
        self.time_to_load_vehicles = time_to_load_vehicles
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport

        self.metrics: List[Dict[str, float]] = []
        if out_csv_name is None:
            self.out_csv_name = None
        else:
            self.out_csv_name = out_csv_name
            if self.out_csv_name.endswith == "/":
                if not os.path.exists(self.out_csv_name):
                    os.makedirs(self.out_csv_name)
            else:
                if not Path(out_csv_name).parent.exists():
                    os.makedirs(str(Path(out_csv_name).parent))

    @property
    def sim_step(self):
        return traci.simulation.getTime()

    def reset(self):
        if self.run != 0:
            traci.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        self.curr_route_file = np.random.choice(self._route_files)

        # Initialize SUMO environments for agents
        sumo_cmd = [
            self._sumo_binary,
            '-n', str(self._net),
            '-r', str(self.curr_route_file),
            '--max-depart-delay', str(self.max_depart_delay),
            '--waiting-time-memory', '10000',
            '--time-to-teleport', str(self.time_to_teleport),
            '--random'
        ]
        if self._additionals_file is not None:
            sumo_cmd.extend(['-a', str(self._additionals_file)])
        if self.use_gui:
            sumo_cmd.append('--start')

        traci.start(sumo_cmd, label=self.agent_id)

        # Build networks for each environment
        self.sumo_net = SumoGridNetwork(self.agent_id, self.num_rows, self.num_cols)
        self.sumo_net.reset()

        return self._compute_observations()

    def step(self, actions: List[int]):
        self.step_num += 1
        if actions is not None:
            self.sumo_net.apply_actions(actions)
        self.sumo_net.step(self.delta_time)

        observations = self._compute_observations()

        beta = min(1.0, max(self.step_num * 1. / self.num_train_steps, 0.0))
        global_reward = self.sumo_net.global_reward()
        local_reward = self.sumo_net.local_reward()
        reward = beta * global_reward + (1 - beta) * local_reward

        info = self.sumo_net.info()
        info["global_reward"] = global_reward
        info["hybrid_reward"] = reward
        info["step_time"] = self.sim_step
        info["current_route_file"] = self.curr_route_file
        self.metrics.append(info)

        dones = self.sim_step > self.sim_max_time

        return observations, reward, dones, info

    def _compute_observations(self):
        return self.sumo_net.as_feature_grid()

    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)

            df.to_csv(
                f"{self.out_csv_name}_agent_id_{self.agent_id}_run_{run}.csv",
                index=False,
            )
