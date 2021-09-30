
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.agent import GridObservingAgent, MovingAgent, GridWorldAgent
from abmarl.sim.gridworld.state import PositionState
from abmarl.sim.gridworld.actor import MoveActor
from abmarl.sim.gridworld.observer import SingleGridObserver
from abmarl.tools.matplotlib_utils import mscatter

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
    
    def render(self, fig=None, **kwargs):
        fig.clear()
        ax = fig.gca()

        # Draw the gridlines
        ax.set(xlim=(0, self.position_state.cols), ylim=(0, self.position_state.rows))
        ax.set_xticks(np.arange(0, self.position_state.cols, 1))
        ax.set_yticks(np.arange(0, self.position_state.rows, 1))
        ax.grid()

        # Draw the agents
        agents_x = [
            agent.position[1] + 0.5 for agent in self.agents.values() if agent.active
        ]
        agents_y = [
            self.position_state.rows - 0.5 - agent.position[0]
            for agent in self.agents.values() if agent.active
        ]
        shape = [agent.render_shape for agent in self.agents.values() if agent.active]
        color = [agent.render_color for agent in self.agents.values() if agent.active]
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, facecolor=color)

        plt.plot()
        plt.pause(1e-6)

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

if __name__ == "__main__":
    object_registry = {
        'N': lambda n: MazeNavigationAgent(
            id=f'navigator',
            encoding=1,
            view_range=2,
            render_color='blue',
        ),
        'T': lambda n: GridWorldAgent(
            id=f'target',
            encoding=3,
            render_color='green'
        ),
        'W': lambda n: GridWorldAgent(
            id=f'wall{n}',
            encoding=2,
            view_blocking=True,
            render_shape='s'
        )
    }

    file_name = 'maze.txt'
    sim = MazeNaviationSim.build_sim_from_file(
        file_name,
        object_registry,
        overlapping={1: [3], 3: [1]}
    )
    sim.reset()
    fig = plt.figure()
    sim.render(fig=fig)

    
    for i in range(100):
        action = {'navigator': sim.navigator.action_space.sample()}
        sim.step(action)
        sim.render(fig=fig)
        done = sim.get_all_done()
        if done:
            plt.pause(1)
            break
