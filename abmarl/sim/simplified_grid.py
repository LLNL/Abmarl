
from gym.spaces import Box
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim import PrincipleAgent, ActingAgent, AgentBasedSimulation
from abmarl.tools.matplotlib_utils import mscatter

class GridAgent(PrincipleAgent):
    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, value):
        self._position = value

class WallAgent(GridAgent): pass

class ExploringAgent(GridAgent, ActingAgent):
    def __init__(self, view_range=None, move_range=None, **kwargs):
        super().__init__(**kwargs)
        self.view_range = view_range
        self.move_range = move_range
        self.action_space = Box(-move_range, move_range, (2,), np.int)




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
        # TODO: distinguish between empty out of bounds and empty in bounds.
        
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

        mask = np.ones((2 * agent.view + 1, 2 * agent.view + 1))

        for other in self.agents.values():
            if isinstance(other, WallAgent):
                x_diff = other.position[1] - agent.position[1]
                y_diff = agent.position[0] - other.position[0]
                if -agent.view <= x_diff <= agent.view and \
                        agent.view <= y_diff <= agent.view and \
                        mask[x_diff + agent.view, y_diff + agent.view] != 0:
                    if x_diff > 0 and y_diff == 0:
                        upper = lambda t: (y_diff + 0.5) / (x_diff - 0.5) * t
                        lower = lambda t: (y_diff - 0.5) / (x_diff - 0.5) * t
                        for x in range(x_diff, agent.view+1):
                            for y in range(-agent.view, agent.view+1):
                                if lower(x) < y < upper(x):
                                    mask[x + agent.view, y + agent.view] = 0
                    elif x_diff > 0 and y_diff > 0:
                        upper = lambda t: (y_diff + 0.5) / (x_diff - 0.5) * t
                        lower = lambda t: (y_diff - 0.5) / (x_diff + 0.5) * t
                        for x in range(x_diff, agent.view+1):
                            for y in range(y_diff, agent.view+1):
                                if lower(x) < y < upper(x):
                                    mask[x + agent.view, y + agent.view] = 0
                    elif x_diff == 0 and y_diff > 0:
                        left = lambda t: (x_diff - 0.5) / (y_diff - 0.5) * t
                        right = lambda t: (x_diff + 0.5) / (y_diff - 0.5) * t
                        for x in range(-agent.view, agent.view+1):
                            for y in range(y_diff, agent.view+1):
                                if left(y) < x < right(y):
                                    mask[x + agent.view, y + agent.view] = 0
                    elif x_diff < 0 and y_diff > 0:
                        upper = lambda t: (y_diff + 0.5) / (x_diff + 0.5) * t
                        lower = lambda t: (y_diff - 0.5) / (x_diff - 0.5) * t
                        for x in range(x_diff, -agent.view-1, -1):
                            for y in range(y_diff, agent.view+1):
                                if lower(x) < y < upper(x):
                                    mask[x + agent.view, y + agent.view] = 0
                    elif x_diff < 0 and y_diff == 0:
                        upper = lambda t: (y_diff + 0.5) / (x_diff + 0.5) * t
                        lower = lambda t: (y_diff - 0.5) / (x_diff + 0.5) * t
                        for x in range(x_diff, -agent.view-1, -1):
                            for y in range(-agent.view, agent.view+1):
                                if lower(x) < y < upper(x):
                                    mask[x + agent.view, y + agent.view] = 0
                    elif x_diff < 0 and y_diff < 0:
                        upper = lambda t: (y_diff + 0.5) / (x_diff - 0.5) * t
                        lower = lambda t: (y_diff - 0.5) / (x_diff + 0.5) * t
                        for x in range(x_diff, -agent.view - 1, -1):
                            for y in range(y_diff, -agent.view - 1, -1):
                                if lower(x) < y < upper(x):
                                    mask[x + agent.view, y + agent.view] = 0
                    elif x_diff == 0 and y_diff < 0:
                        left = lambda t: (x_diff - 0.5) / (y_diff + 0.5) * t
                        right = lambda t: (x_diff + 0.5) / (y_diff + 0.5) * t
                        for x in range(-agent.view, agent.view+1):
                            for y in range(y_diff, -agent.view - 1, -1):
                                if left(y) < x < right(y):
                                    mask[x + agent.view, y + agent.view] = 0
                    elif x_diff > 0 and y_diff < 0:
                        upper = lambda t: (y_diff + 0.5) / (x_diff + 0.5) * t
                        lower = lambda t: (y_diff - 0.5) / (x_diff - 0.5) * t
                        for x in range(x_diff, agent.view+1):
                            for y in range(y_diff, -agent.view - 1, -1):
                                if lower(x) < y < upper(x):
                                    mask[x + agent.view, y + agent.view] = 0
                    
                    mask[x_diff + agent.view, y_diff + agent.view] = 2

        mask = np.flipud(np.transpose(mask))




        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)
        # --- end gen_obs_grid()

        # --- grid.encode
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()

        image = array
        # --- end grid.encode

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
    f'explorer{i}': ExploringAgent(id=f'explorer{i}', move_range=1) for i in range(5)
}
walls = {
    f'wall{i}': WallAgent(id=f'wall{i}') for i in range(12)
}
agents = {**explorers, **walls}
sim = GridSim(rows=8, cols=12, agents=agents)
sim.reset()
sim.render(fig=fig)

for _ in range(100):
    action = {agent.id: agent.action_space.sample() for agent in agents.values() if isinstance(agent, ActingAgent)}
    import pprint; pprint.pprint(action)
    sim.step(action)
    sim.render(fig=fig)
