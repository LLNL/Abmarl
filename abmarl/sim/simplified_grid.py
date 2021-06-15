
from gym.spaces import Box
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim import PrincipleAgent, Agent, ActingAgent, AgentBasedSimulation
from abmarl.tools.matplotlib_utils import mscatter

class GridAgent(PrincipleAgent):
    def __init__(self, encode=None, **kwargs):
        super().__init__(**kwargs)
        self.encode = encode

    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, value):
        self._position = value
    
    @property
    def encode(self):
        return self._encode
    
    @encode.setter
    def encode(self, value):
        assert type(value) is int, f"{self.id}'s encoding must be an integer."
        assert value != -2, "-2 encoding reserved for masked observation."
        assert value != -1, "-1 encoding reserved for out of bounds."
        assert value != 0, "0 encoding reserved for empty cell."
        self._encode = value

class WallAgent(GridAgent):
    def __init__(self, encode=2, **kwargs):
        super().__init__(**{'encode': encode, **kwargs})

class ExploringAgent(GridAgent, Agent):
    def __init__(self, encode=1, view_range=None, move_range=None, **kwargs):
        super().__init__(**{'encode': encode, **kwargs})
        self.view_range = view_range
        self.move_range = move_range
        self.action_space = Box(-move_range, move_range, (2,), np.int)
        self.observation_space = Box(-np.inf, np.inf, (view_range, view_range), np.int)


class GridSim(AgentBasedSimulation):
    def __init__(self, rows=None, cols=None, agents=None, **kwargs):
        self.rows = rows
        self.cols = cols
        # Dictionary lookup by id
        self.agents = agents
        # Grid lookup by position
        self.grid = np.empty((rows, cols), dtype=object)
        
        self.finalize()
    
    def reset(self, **kwargs):
        # Choose unique positions in the grid
        rs, cs = np.unravel_index(
            np.random.choice(self.rows * self.cols, len(self.agents), False),
            shape=(self.rows, self.cols)
        )
        for ndx, agent in enumerate(self.agents.values()): # Assuming all agents are GridAgent
            r = rs[ndx]
            c = cs[ndx]
            agent.position = np.array([r, c])
            self.grid[r, c] = agent
    
    def step(self, action_dict):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            new_position = agent.position + action
            if 0 <= new_position[0] < self.rows and 0 <= new_position[1] < self.cols and self.grid[new_position[0], new_position[1]] is None:
                self.grid[agent.position[0], agent.position[1]] = None
                agent.position = new_position
                self.grid[agent.position[0], agent.position[1]] = agent

    def render(self, fig=None, **kwargs):
        fig.clear()

        # Draw the resources
        ax = fig.gca()

        # Draw the agents
        shape_dict = {
            agent.id: 'o' if isinstance(agent, ExploringAgent) else 's'
            for agent in self.agents.values()
        }
        ax.set(xlim=(0, self.cols), ylim=(0, self.rows))
        ax.set_xticks(np.arange(0, self.cols, 1))
        ax.set_yticks(np.arange(0, self.rows, 1))
        ax.grid()

        agents_x = [
            agent.position[1] + 0.5 for agent in self.agents.values()
        ]
        agents_y = [
            self.rows - 0.5 - agent.position[0] for agent in self.agents.values()
        ]
        shape = [shape_dict[agent_id] for agent_id in shape_dict]
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)

    def get_obs(self, agent_id, **kwargs):
        """
        Return the agent's observation.
        """
        agent = self.agents[agent_id]
        # Generate a completely empty grid
        local_grid = np.empty((agent.view_range * 2 + 1, agent.view_range * 2 + 1), dtype=object)
        local_grid.fill(-1)
        
        # Copy the section of the grid around the agent's location
        (r, c) = agent.position
        r_lower = max([0, r - agent.view_range])
        r_upper = min([self.rows - 1, r + agent.view_range]) + 1
        c_lower = max([0, c - agent.view_range])
        c_upper = min([self.cols - 1, c+agent.view_range]) + 1
        local_grid[
            (r_lower+agent.view_range-r):(r_upper+agent.view_range-r),
            (c_lower+agent.view_range-c):(c_upper+agent.view_range-c)
        ] = self.grid[r_lower:r_upper, c_lower:c_upper]

        # Generate an observation mask. The agent's observation can be blocked
        # by walls, which hide the cells "behind" them. In the mask, 1 means that
        # this square is visibile, 0 means that it is invisible.
        mask = np.ones((2 * agent.view_range + 1, 2 * agent.view_range + 1))
        for other in self.agents.values():
            if isinstance(other, WallAgent):
                r_diff, c_diff = other.position - agent.position
                if -agent.view_range <= r_diff <= agent.view_range and \
                        -agent.view_range <= c_diff <= agent.view_range:
                    if c_diff > 0 and r_diff == 0: # Wall is to the right of agent
                        upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
                        lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
                        for c in range(c_diff, agent.view_range+1):
                            for r in range(-agent.view_range, agent.view_range+1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if lower(c) < r < upper(c):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff > 0 and r_diff > 0: # Wall is below-right of agent
                        upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
                        lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
                        for c in range(c_diff, agent.view_range+1):
                            for r in range(r_diff, agent.view_range+1):
                                if c == c_diff and r == r_diff: continue # Don't mask the wall
                                if lower(c) < r < upper(c):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff == 0 and r_diff > 0: # Wall is below the agent
                        left = lambda t: (c_diff - 0.5) / (r_diff - 0.5) * t
                        right = lambda t: (c_diff + 0.5) / (r_diff - 0.5) * t
                        for c in range(-agent.view_range, agent.view_range+1):
                            for r in range(r_diff, agent.view_range+1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if left(r) < c < right(r):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff < 0 and r_diff > 0: # Wall is below-left of agent
                        upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
                        lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
                        for c in range(c_diff, -agent.view_range-1, -1):
                            for r in range(r_diff, agent.view_range+1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if lower(c) < r < upper(c):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff < 0 and r_diff == 0: # Wall is left of agent
                        upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
                        lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
                        for c in range(c_diff, -agent.view_range-1, -1):
                            for r in range(-agent.view_range, agent.view_range+1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if lower(c) < r < upper(c):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff < 0 and r_diff < 0: # Wall is above-left of agent
                        upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
                        lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
                        for c in range(c_diff, -agent.view_range - 1, -1):
                            for r in range(r_diff, -agent.view_range - 1, -1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if lower(c) < r < upper(c):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff == 0 and r_diff < 0: # Wall is above the agent
                        left = lambda t: (c_diff - 0.5) / (r_diff + 0.5) * t
                        right = lambda t: (c_diff + 0.5) / (r_diff + 0.5) * t
                        for c in range(-agent.view_range, agent.view_range+1):
                            for r in range(r_diff, -agent.view_range - 1, -1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if left(r) < c < right(r):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff > 0 and r_diff < 0: # Wall is above-right of agent
                        upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
                        lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
                        for c in range(c_diff, agent.view_range+1):
                            for r in range(r_diff, -agent.view_range - 1, -1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if lower(c) < r < upper(c):
                                    mask[r + agent.view_range, c + agent.view_range] = 0

        # Convolve the grid observation with the mask.
        obs = np.zeros((2 * agent.view_range + 1, 2 * agent.view_range + 1), dtype=np.int)
        for r in range(2 * agent.view_range + 1):
            for c in range(2 * agent.view_range + 1):
                if mask[r, c]:
                    obj = local_grid[r, c]
                    if obj == -1: # Out of bounds
                        obs[r, c] = -1
                    elif obj is None: # Empty
                        obs[r, c] = 0
                    else: # Something there, so get its encoding
                        obs[r, c] = obj.encode
                else: # Cell blocked by wall. Indicate invisible with -2
                    obs[r, c] = -2

        return obs

    def get_reward(self, agent_id, **kwargs):
        """
        Return the agent's reward.
        """
        pass

    def get_done(self, agent_id, **kwargs):
        """
        Return the agent's done status.
        """
        pass

    def get_all_done(self, **kwargs):
        """
        Return the simulation's done status.
        """
        pass

    def get_info(self, agent_id, **kwargs):
        """
        Return the agent's info.
        """
        pass

fig = plt.figure()
explorers = {
    f'explorer{i}': ExploringAgent(id=f'explorer{i}', move_range=1, view_range=3) for i in range(5)
}
explorers['explorer0'].encode = 5
walls = {
    f'wall{i}': WallAgent(id=f'wall{i}') for i in range(12)
}
agents = {**explorers, **walls}
sim = GridSim(rows=8, cols=12, agents=agents)
sim.reset()
sim.render(fig=fig)

# Agents move around
for _ in range(100):
    action = {agent.id: agent.action_space.sample() for agent in agents.values() if isinstance(agent, ActingAgent)}
    sim.step(action)
    sim.render(fig=fig)

# Examine the agents' observations
from pprint import pprint
for agent in explorers.values():
    print(agent.position)
    pprint(sim.get_obs(agent.id))
    print()

plt.show()
