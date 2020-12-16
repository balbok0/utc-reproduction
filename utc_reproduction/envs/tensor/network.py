import traci
from .traffic_signal import TrafficSignal
import numpy as np
from typing import List, Dict, Any


class SumoTensorNetwork:
    def __init__(self, label: int, num_rows: int, num_cols: int):
        self.label = label
        self.num_rows = num_rows
        self.num_cols = num_cols

        traci.switch(self.label)
        self.ts_ids = traci.trafficlight.getIDList()
        self.traffic_signals = {
            ts_id: TrafficSignal(ts_id)
            for ts_id in self.ts_ids
        }

        self.ts_ids = sorted(
            self.ts_ids,
            # Left to right/Top to bottom
            key=lambda x: (-self.traffic_signals[x].position[1], self.traffic_signals[x].position[0]),
        )

        self.last_step = 0
        self.outgoing: List[int] = []
        self.incoming: List[int] = []

    @property
    def veh_out(self):
        return sum(self.outgoing[-self.last_step:]) / self.last_step

    @property
    def veh_in(self):
        return sum(self.incoming[-self.last_step:]) / self.last_step

    def reset(self):
        self.step_num = 0

    def as_feature_grid(self):
        traci.switch(self.label)
        observations = np.zeros((2, 4 * self.num_rows, 4 * self.num_cols))
        for ts_idx, ts in enumerate(self.ts_ids):
            r_i = ts_idx // self.num_cols
            c_i = ts_idx % self.num_cols
            observations[:, r_i * 4:(r_i + 1) * 4, c_i * 4:(c_i + 1) * 4] = self.traffic_signals[ts].as_feature_grid()
        return observations

    def apply_actions(self, actions: np.ndarray):
        traci.switch(self.label)
        for ts_id, action in zip(self.ts_ids, actions):
            if action == 1:
                self.traffic_signals[ts_id].iter_phase()

    def step(self, step: int = 1):
        assert step > 0
        self.last_step = step
        traci.switch(self.label)
        for _ in range(step):
            traci.simulationStep()
            arrived_ids = set(traci.simulation.getArrivedIDList())
            teleported_ids = set(traci.simulation.getEndingTeleportIDList())
            self.outgoing.append(len(arrived_ids - teleported_ids))
            self.incoming.append(traci.simulation.getDepartedNumber())

    def global_reward(self):
        traci.switch(self.label)
        global_reward = self.veh_out - self.veh_in
        return global_reward

    def local_reward(self):
        traci.switch(self.label)
        local_reward = sum([ts.reward() for ts in self.traffic_signals.values()]) / len(self.ts_ids)

        return local_reward

    def info(self) -> Dict[str, Any]:
        return {
            "avg_wait_time": self.avg_waiting_time(),
            "avg_stopped": self.avg_stopped(),
        }
        pass

    def avg_stopped(self) -> float:
        num_stopped = 1.0 * sum([sum(ts.get_queues()) for ts in self.traffic_signals.values()])
        return num_stopped / len(self.ts_ids)

    def avg_waiting_time(self) -> float:
        sum_wait = 0
        sum_vehicles = 0

        for ts in self.traffic_signals.values():
            sum_wait += ts.get_total_waiting_time()
            sum_vehicles += len(ts.vehicles_in)

        if sum_vehicles == 0:
            return 0.0

        return sum_wait / sum_vehicles
