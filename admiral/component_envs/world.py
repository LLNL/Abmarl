
from abc import ABC, abstractmethod

import numpy as np

from admiral.envs import Agent

class WorldAgent(Agent):
    """
    WorldAgents have a position in the world. This position can given to the agent
    as a starting position for each episode. If the position is None, then the environment
    should assign the agent a random position.
    """
    def __init__(self, starting_position=None, **kwargs):
        super().__init__(**kwargs)
        self.starting_position = starting_position
    
    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return super().configured and self.starting_position is not None

class WorldEnv(ABC):
    """
    WorldEnv is an abstract notion for some space in which agents exist. It is defined
    by the set of agents that exist in it and the bounds of the world. Agents in
    this environment will have a position at reset.
    """
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.region = region
        self.agents = agents if agents is not None else {}
        for agent in self.agents.values():
            assert isinstance(agent, WorldAgent)

    def reset(self, **kwargs):
        """
        Place agents throughout the world.
        """
        for agent in self.agents.values():
            if agent.starting_position is not None:
                agent.position = agent.starting_position
            else:
                self._assign_position(agent, **kwargs)
        
    @abstractmethod
    def _assign_position(self, agent, **kwargs):
        """Randomly assign the agent a position in the world."""
        pass

    @abstractmethod
    def render(self, **kwargs):
        """
        Draw the world with the agents.
        """
        pass

class ContinuousWorldEnv(WorldEnv):
    """
    Skeleton class for what a continuous world might look like.
    """
    def _assign_position(self, agent, **kwargs):
        agent.position = np.random.uniform(0, self.region, 2)

class GridWorldEnv(WorldEnv):
    """
    Grid world is made up of region x region cells.
    """
    def _assign_position(self, agent, **kwargs):
        """
        Randomly place the agents on cells in the grid.
        """
        agent.position = np.random.randint(0, self.region, 2)

    def render(self, fig=None, render_condition=None, **kwargs):
        """
        Draw the agents are gray circles in the grid.
        """
        draw_now = fig is None
        if draw_now:
            from matplotlib import pyplot as plt
            fig = plt.gcf()

        ax = fig.gca()
        ax.set(xlim=(0, self.region), ylim=(0, self.region))
        ax.set_xticks(np.arange(0, self.region, 1))
        ax.set_yticks(np.arange(0, self.region, 1))
        ax.grid()

        if render_condition is None:
            agents_x = [agent.position[1] + 0.5 for agent in self.agents.values()]
            agents_y = [self.region - 0.5 - agent.position[0] for agent in self.agents.values()]
        else:
            agents_x = [agent.position[1] + 0.5 for agent in self.agents.values() if render_condition[agent.id]]
            agents_y = [self.region - 0.5 - agent.position[0] for agent in self.agents.values() if render_condition[agent.id]]

        ax.scatter(agents_x, agents_y, marker='o', s=200,  edgecolor='black', facecolor='gray')

        if draw_now:
            plt.plot()
            plt.pause(1e-17)

        return ax
