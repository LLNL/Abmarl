
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.agent import GridWorldAgent
from abmarl.sim.gridworld.state import PositionState

class BlankSpace(GridWorldAgent):
    def __init__(self, **kwargs):
        super().__init__(
            encoding=1,
            render_color='gray',
            render_shape='s',
            **kwargs
        )
class Apple(GridWorldAgent):
    def __init__(self, **kwargs):
        super().__init__(
            encoding=2,
            render_color='green',
            **kwargs
        )
class Peach(GridWorldAgent):
    def __init__(self, **kwargs):
        super().__init__(
            encoding=3,
            render_color='orange',
            **kwargs
        )
class Pear(GridWorldAgent):
    def __init__(self, **kwargs):
        super().__init__(
            encoding=4,
            render_color='yellow',
            **kwargs
        )
class Plum(GridWorldAgent):
    def __init__(self, **kwargs):
        super().__init__(
            encoding=5,
            render_color='purple',
            **kwargs
        )
class Cherry(GridWorldAgent):
    def __init__(self, **kwargs):
        super().__init__(
            encoding=6,
            render_color='red',
            **kwargs
        )

class DiversifySim(GridWorldSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        self.position_state = PositionState(**kwargs)
        
        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.reward = 0

    def step(self, action_dict, **kwargs):
        pass

    def get_obs(self, agent_id, **kwargs):
        return {}
    
    def get_reward(self, *args, **kwargs):
        return 0
    
    def get_done(self, agent_id, **kwargs):
        return True
    
    def get_all_done(self, **kwargs):
        return True
    
    def get_info(self, agent_id, **kwargs):
        return {}
