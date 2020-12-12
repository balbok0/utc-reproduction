# Adapted from: https://github.com/LucasAlegre/sumo-rl

import traci
import numpy as np


class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """
    def __init__(
        self,
        ts_id: str,
        strict=True,
    ):
        self.id = ts_id

        _junction = traci.trafficlight._getUniversal(traci.constants.TL_CONTROLLED_JUNCTIONS, self.id)
        if len(_junction) > 1 and strict:
            raise EnvironmentError(
                f"There are {len(_junction)} junctions with ids: {_junction}, however TrafficSignal can process only one junction per traffic signal.\n"
                f"To avoid it either create multiple traffic signals in place of current one (id: {self.id}),\n"
                f"  or set `strict=True` (in this case first junction will be considered)"
            )
        self.junction = _junction[0]

        self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))  # remove duplicates and keep order
        self.in_lanes = [link[0][0] for link in traci.trafficlight.getControlledLinks(self.id)]
        self.in_lanes = list(dict.fromkeys(self.in_lanes))  # remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in traci.trafficlight.getControlledLinks(self.id)]
        self.out_lanes = list(dict.fromkeys(self.out_lanes))

        program_id = traci.trafficlight.getProgram(ts_id)
        logics = traci.trafficlight.getAllProgramLogics(ts_id)

        logic = [logic for logic in logics if logic.programID == program_id][0]
        self.phases = logic.getPhases()

        self._lane_to_e2_detector = {}
        for e2_id in traci.lanearea.getIDList():
            self._lane_to_e2_detector[traci.lanearea.getLaneID(e2_id)] = e2_id

    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    @property
    def position(self):
        return traci.junction.getPosition(self.junction)

    @property
    def vehicles_in(self):
        result = []
        for lane in self.in_lanes:
            if lane in self._lane_to_e2_detector:
                result.extend(traci.lanearea.getLastStepVehicleIDs(self._lane_to_e2_detector[lane]))
            else:
                result.extend(traci.lane.getLastStepVehicleIDs(lane))
        return result

    def iter_phase(self):
        # If it contains a yellow light, then ignore iteration.
        if "y" in self.phases[self.phase].state:
            return

        new_phase = (self.phase + 1) % len(self.phases)
        traci.trafficlight.setPhase(self.id, new_phase)

    def get_mean_speeds(self):
        result = np.zeros(len(self.in_lanes))
        for lane_idx, lane in enumerate(self.in_lanes):
            if lane in self._lane_to_e2_detector:
                result[lane_idx] = traci.lanearea.getLastStepMeanSpeed(self._lane_to_e2_detector[lane])
            else:
                result[lane_idx] = traci.lane.getLastStepMeanSpeed(lane)

        # Traci likes -1
        result[result == -1] = 0

        return result

    def get_total_mean_speed(self):
        result = 0.0
        count = 0
        for lane in self.in_lanes:
            if lane in self._lane_to_e2_detector:
                lane_mean = traci.lanearea.getLastStepMeanSpeed(self._lane_to_e2_detector[lane])
                lane_count = traci.lanearea.getLastStepVehicleNumber(self._lane_to_e2_detector[lane])
            else:
                lane_mean = traci.lane.getLastStepMeanSpeed(lane)
                lane_count = traci.lane.getLastStepVehicleNumber(lane)
            if result != -1:
                result += lane_count * lane_mean
                count += lane_count
        if count == 0:
            return 0.0
        else:
            return result / count

    def get_queues(self):
        result = np.zeros(len(self.in_lanes))
        for lane_idx, lane in enumerate(self.in_lanes):
            if lane in self._lane_to_e2_detector:
                result[lane_idx] = traci.lanearea.getLastStepHaltingNumber(self._lane_to_e2_detector[lane])
            else:
                result[lane_idx] = traci.lane.getLastStepHaltingNumber(lane)

        # Traci likes -1
        result[result == -1] = 0
        return result

    def get_total_waiting_time(self):
        result = 0.0
        for vehicle in self.vehicles_in:
            result += traci.vehicle.getWaitingTime(vehicle)
        return result

    def as_feature_grid(self):
        # Lanes seem to always be top, right, bottom, left
        idx_to_loc = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 2), (3, 1), (2, 0), (1, 0)]
        grid = np.zeros((2, 4, 4))
        queues = self.get_queues()
        mean_speeds = self.get_mean_speeds()
        for (x, y), lane_queue, lane_mean_speed in zip(idx_to_loc, queues, mean_speeds):
            grid[0, x, y] = lane_queue
            grid[1, x, y] = lane_mean_speed
        print(grid)
        return grid

    def reward(self):
        num_halted_southbound = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.in_lanes[:2]])
        num_halted_westbound = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.in_lanes[2:4]])
        num_halted_northbound = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.in_lanes[4:6]])
        num_halted_eastbound = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.in_lanes[-2:]])
        return -abs(
            max(num_halted_northbound, num_halted_southbound) - max(num_halted_eastbound, num_halted_westbound)
        )
