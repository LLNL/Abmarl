
from abc import ABC, abstractmethod

import numpy as np

from admiral.envs import Agent
from admiral.tools.matplotlib_utils import mscatter

class GridWorldAgent(Agent):
    """
    WorldAgents have a position in the world. This position can given to the agent
    as a starting position for each episode. If the position is None, then the environment
    should assign the agent a random position.
    """
    def __init__(self, view=None, starting_position=None, **kwargs):
        super().__init__(**kwargs)
        assert view is not None, "GridWorldAgent must have a view"
        self.view = view
        self.starting_position = starting_position
    
    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return super().configured and self.view is not None

class GridWorldEnv(ABC):
    """
    WorldEnv is an abstract notion for some space in which agents exist. It is defined
    by the set of agents that exist in it and the bounds of the world. Agents in
    this environment will have a position at reset.
    """
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.region = region
        assert type(agents) is dict, "agents must be a dict"
        for agent in agents.values():
            assert isinstance(agent, GridWorldAgent)
        self.agents = agents

        from gym.spaces import Box
        for agent in self.agents.values():
            agent.observation_space['agents'] = Box(-1, 1, (agent.view*2+1, agent.view*2+1), np.int)

    def reset(self, **kwargs):
        """
        Place agents throughout the world.
        """
        for agent in self.agents.values():
            if agent.starting_position is not None:
                agent.position = agent.starting_position
            else:
                agent.position = np.random.randint(0, self.region, 2)

    def render(self, fig=None, render_condition={}, shape_dict={}, **kwargs):
        """
        Draw the agents in the grid. The shape of each agent is dictated by shape_dict.
        If that is empty, then the agents are drawn as circles.
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

        if render_condition:
            agents_x = [agent.position[1] + 0.5 for agent in self.agents.values() if render_condition[agent.id]]
            agents_y = [self.region - 0.5 - agent.position[0] for agent in self.agents.values() if render_condition[agent.id]]
        else:
            agents_x = [agent.position[1] + 0.5 for agent in self.agents.values()]
            agents_y = [self.region - 0.5 - agent.position[0] for agent in self.agents.values()]
            render_condition = {agent_id: True for agent_id in self.agents}

        if shape_dict:
            shape = [shape_dict[agent_id] for agent_id in shape_dict if render_condition[agent_id]]
        else:
            shape = 'o'
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, edgecolor='black', facecolor='gray')

        if draw_now:
            plt.plot()
            plt.pause(1e-17)

        return ax
    
    def get_obs(self, my_id, **kwargs):
        my_agent = self.agents[my_id]
        signal = np.zeros((my_agent.view*2+1, my_agent.view*2+1))

        # --- Determine the boundaries of the agents' grids --- #
        # For left and top, we just do: view - x,y >= 0
        # For the right and bottom, we just do region - x,y - 1 - view > 0
        if my_agent.view - my_agent.position[0] >= 0: # Top end
            signal[0:my_agent.view - my_agent.position[0], :] = -1
        if my_agent.view - my_agent.position[1] >= 0: # Left end
            signal[:, 0:my_agent.view - my_agent.position[1]] = -1
        if self.region - my_agent.position[0] - my_agent.view - 1 < 0: # Bottom end
            signal[self.region - my_agent.position[0] - my_agent.view - 1:,:] = -1
        if self.region - my_agent.position[1] - my_agent.view - 1 < 0: # Right end
            signal[:, self.region - my_agent.position[1] - my_agent.view - 1:] = -1

        # --- Determine the positions of all the other alive agents --- #
        for other_id, other_agent in self.agents.items():
            if other_id == my_id: continue # Don't observe yourself
            r_diff = other_agent.position[0] - my_agent.position[0]
            c_diff = other_agent.position[1] - my_agent.position[1]
            if -my_agent.view <= r_diff <= my_agent.view and -my_agent.view <= c_diff <= my_agent.view:
                r_diff += my_agent.view
                c_diff += my_agent.view
                signal[r_diff, c_diff] = 1 # There is an agent at this location.

        return signal

