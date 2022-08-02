
import numpy as np

from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.agent import GridObservingAgent, MovingAgent
from abmarl.sim.gridworld.state import PositionState
from abmarl.sim.gridworld.actor import MoveActor
from abmarl.sim.gridworld.observer import SingleGridObserver


class MazeNavigationAgent(GridObservingAgent, MovingAgent):
    def __init__(self, **kwargs):
        super().__init__(move_range=1, **kwargs)


class MazeNaviationSim(GridWorldSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.navigator = kwargs['agents']['navigator']
        self.target = kwargs['agents']['target']

        # State Components
        self.position_state = PositionState(**kwargs)

        # Action Components
        self.move_actor = MoveActor(**kwargs)

        # Observation Components
        self.grid_observer = SingleGridObserver(**kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)

        # Track the rewards
        self.reward = 0

    def step(self, action_dict, **kwargs):
        # Process moves
        action = action_dict['navigator']
        move_result = self.move_actor.process_action(self.navigator, action, **kwargs)
        if not move_result:
            self.reward -= 0.1

        # Entropy penalty
        self.reward -= 0.01

    def get_obs(self, agent_id, **kwargs):
        return {
            **self.grid_observer.get_obs(self.navigator, **kwargs)
        }

    def get_reward(self, agent_id, **kwargs):
        if self.get_all_done():
            self.reward = 1
        reward = self.reward
        self.reward = 0
        return reward

    def get_done(self, agent_id, **kwargs):
        return self.get_all_done()

    def get_all_done(self, **kwargs):
        return np.all(self.navigator.position == self.target.position)

    def get_info(self, agent_id, **kwargs):
        return {}
