
from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.state import PositionState


class MultiAgentGridSim(GridWorldSimulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.position_state = PositionState(**kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset()

    def step(self, action_dict, **kwargs):
        pass

    def get_obs(self, agent_id, **kwargs):
        return {}

    def get_reward(self, agent_id, **kwargs):
        return 0

    def get_done(self, agent_id, **kwargs):
        return False

    def get_all_done(self, **kwargs):
        return False

    def get_info(self, agent_id, **kwargs):
        return {}
