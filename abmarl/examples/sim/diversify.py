
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.agent import GridWorldAgent
from abmarl.sim.gridworld.state import PositionState
import abmarl.sim.gridworld.utils as gu

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
class FutureTree(GridWorldAgent):
    def __init__(self, **kwargs):
        super().__init__(
            encoding=7,
            render_color='brown',
            render_shape='x',
            **kwargs
        )

class DiversifySim(GridWorldSimulation):
    def __init__(self, reward_type='neighbor', **kwargs):
        self.agents = kwargs['agents']
        self.grid = kwargs['grid']

        self.position_state = PositionState(**kwargs)

        self.get_reward = {
            'neighbor': self._get_reward_neighbors,
            'neighbor2': self._get_reward_neighbors_2,
            'distance': self._get_reward_distance,
        }[reward_type]
        
        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.reward = 0

    def step(self, action_dict, **kwargs):
        pass

    def get_obs(self, agent_id, **kwargs):
        return {}
    
    def get_reward(self, *args, **kwargs):
        pass
    
    def _get_reward_neighbors(self, *args, **kwargs):
        rewards = {
            agent.id: 0 for agent in self.agents.values()
            if isinstance(agent, (Apple, Peach, Pear, Plum, Cherry, FutureTree))
        }
        for agent in self.agents.values():
            if isinstance(agent, (Apple, Peach, Pear, Plum, Cherry, FutureTree)):
                local_grid, _ = gu.create_grid_and_mask(
                    agent, self.grid, 1, self.agents
                )
                for candidate_agent in [
                    local_grid[0, 0],
                    local_grid[2, 0],
                    local_grid[0, 2],
                    local_grid[2, 2],
                ]:
                    if candidate_agent:
                        candidate_agent = next(iter(candidate_agent.values()))
                        if candidate_agent != agent and candidate_agent.encoding == agent.encoding:
                            rewards[agent.id] += 1
        
        return np.mean([*rewards.values()])
    
    def _get_reward_neighbors_2(self, *args, **kwargs):
        rewards = {
            agent.id: 0 for agent in self.agents.values()
            if isinstance(agent, (Apple, Peach, Pear, Plum, Cherry, FutureTree))
        }
        for agent in self.agents.values():
            if isinstance(agent, (Apple, Peach, Pear, Plum, Cherry, FutureTree)):
                local_grid, _ = gu.create_grid_and_mask(
                    agent, self.grid, 2, self.agents
                )
                for candidate_agent in [
                    local_grid[1, 1], # 17 ft away
                    local_grid[3, 1], # 17 ft away
                    local_grid[1, 3], # 17 ft away
                    local_grid[0, 2], # 20 ft away
                    local_grid[4, 2], # 20 ft away
                ]:
                    if candidate_agent:
                        candidate_agent = next(iter(candidate_agent.values()))
                        if candidate_agent.encoding == agent.encoding:
                            rewards[agent.id] += 1
        
        return np.mean([*rewards.values()])
    
    def _get_reward_distance(self, *args, **kwargs):
        rewards = {
            agent.id: 0 for agent in self.agents.values()
            if isinstance(agent, (Apple, Peach, Pear, Plum, Cherry, FutureTree))
        }
        for agent in self.agents.values():
            if not isinstance(agent, (Apple, Peach, Pear, Plum, Cherry, FutureTree)):
                continue
            for other in self.agents.values():
                if other == agent: continue # Don't need because distance is 0
                if other.encoding == agent.encoding:
                    rewards[agent.id] += np.linalg.norm(agent.position - other.position)
        
        return np.mean([*rewards.values()])
    
    def get_done(self, agent_id, **kwargs):
        return True
    
    def get_all_done(self, **kwargs):
        return True
    
    def get_info(self, agent_id, **kwargs):
        return {}
