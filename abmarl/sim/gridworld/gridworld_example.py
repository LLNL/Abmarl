
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim import AgentBasedSimulation
from abmarl.sim.gridworld import GridWorldAgent, GridObservingAgent, MovingAgent
from abmarl.sim.gridworld.state import GridWorldState
from abmarl.sim.gridworld.actor import MoveActor
from abmarl.sim.gridworld.observer import GridObserver
from abmarl.tools.matplotlib_utils import mscatter


class WallAgent(GridWorldAgent):
    """
    Wall agents, immobile and view blocking.

    Args:
        encoding: Default encoding is 1.
    """
    def __init__(self, encoding=1, **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding


class ExploringAgent(MovingAgent, GridObservingAgent):
    """
    Exploring agents, moving around and observing the grid.

    Args:
        encoding: Default encoding is 2.
    """
    def __init__(self, encoding=2, render_shape='o', **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.render_shape = render_shape


class GridSim(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State Components
        self.grid_state = GridWorldState(**kwargs)

        # Action Components
        self.move_actor = MoveActor(grid_state=self.grid_state, **kwargs)

        # Observation Components
        self.grid_observer = GridObserver(grid_state=self.grid_state, **kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.grid_state.reset(**kwargs)

    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            self.move_actor.process_action(agent, action, **kwargs)

    def render(self, fig=None, **kwargs):
        fig.clear()
        ax = fig.gca()

        # Draw the gridlines
        ax.set(xlim=(0, self.grid_state.cols), ylim=(0, self.grid_state.rows))
        ax.set_xticks(np.arange(0, self.grid_state.cols, 1))
        ax.set_yticks(np.arange(0, self.grid_state.rows, 1))
        ax.grid()
        # Draw the agents
        agents_x = [
            agent.position[1] + 0.5 for agent in self.agents.values()
        ]
        agents_y = [
            self.grid_state.rows - 0.5 - agent.position[0] for agent in self.agents.values()
        ]
        shape = [agent.render_shape for agent in self.agents.values()]
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)

    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return {
            **self.grid_observer.get_obs(agent, **kwargs)
        }

    def get_reward(self, agent_id, **kwargs):
        pass

    def get_done(self, agent_id, **kwargs):
        pass

    def get_all_done(self, **kwargs):
        pass

    def get_info(self, agent_id, **kwargs):
        pass


if __name__ == "__main__":
    from abmarl.sim import ActingAgent

    fig = plt.figure()
    explorers = {
        f'explorer{i}': ExploringAgent(id=f'explorer{i}', move_range=1, view_range=3)
        for i in range(5)
    }
    explorers['explorer0'].encoding = 5
    walls = {
        f'wall{i}': WallAgent(id=f'wall{i}', view_blocking=True) for i in range(12)
    }
    agents = {**explorers, **walls}
    sim = GridSim(rows=8, cols=12, agents=agents)
    sim.reset()
    sim.render(fig=fig)

    # Agents move around
    for _ in range(50):
        action = {
            agent.id: agent.action_space.sample() for agent in agents.values()
            if isinstance(agent, ActingAgent)
        }
        sim.step(action)
        sim.render(fig=fig)

    # Examine the agents' observations
    from pprint import pprint
    for agent in explorers.values():
        print(agent.position)
        pprint(sim.get_obs(agent.id)['grid'])
        print()

    # Ensure proper observation space
    for agent in explorers.values():
        print(agent.observation_space)

    plt.show()
