
import numpy as np

from abmarl.sim.gridworld.smart import SmartGridWorldSimulation
from abmarl.sim.gridworld.agent import GridObservingAgent, MovingAgent
from abmarl.sim.gridworld.actor import MoveActor


class MazeNavigationAgent(GridObservingAgent, MovingAgent):
    def __init__(self, **kwargs):
        super().__init__(move_range=1, **kwargs)


class MazeNavigationSim(SmartGridWorldSimulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.navigator = self.agents['navigator']
        self.target = self.agents['target']

        # Action Components
        self.move_actor = MoveActor(**kwargs)

        self.finalize()

    def step(self, action_dict, **kwargs):
        # Process moves
        action = action_dict['navigator']
        move_result = self.move_actor.process_action(self.navigator, action, **kwargs)
        if not move_result:
            self.rewards['navigator'] -= 0.1

        if self.get_all_done():
            self.rewards['navigator'] += 1

        # Entropy penalty
        self.rewards['navigator'] -= 0.01

    def get_done(self, agent_id, **kwargs):
        return self.get_all_done()

    def get_all_done(self, **kwargs):
        return np.all(self.navigator.position == self.target.position)
