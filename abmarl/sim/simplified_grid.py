
from gym.spaces import Box
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim import PrincipleAgent, ActingAgent, AgentBasedSimulation

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
        plt.scatter(agents_x, agents_y, s=200, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)

    def get_obs(self, agent_id, **kwargs):
        """
        Return the agent's observation.
        """
        pass

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
exploring_agents = {
    f'agent{i}': ExploringAgent(id=f'agent{i}', move_range=1) for i in range(5)
}
sim = GridSim(rows=8, cols=12, agents=exploring_agents)
sim.reset()
sim.render(fig=fig)

for _ in range(50):
    action = {agent.id: agent.action_space.sample() for agent in exploring_agents.values()}
    import pprint; pprint.pprint(action)
    sim.step(action)
    sim.render(fig=fig)
