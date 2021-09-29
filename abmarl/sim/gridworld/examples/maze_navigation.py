
from abmarl.sim.gridworld.grid import Grid
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.agent import GridObservingAgent, MovingAgent, GridWorldAgent, AttackingAgent
from abmarl.sim.gridworld.state import PositionState
from abmarl.sim.gridworld.actor import MoveActor
from abmarl.sim.gridworld.observer import SingleGridObserver
from abmarl.tools.matplotlib_utils import mscatter

class MazeNavigationAgent(GridObservingAgent, MovingAgent, AttackingAgent):
    def __init__(self, **kwargs):
        super().__init__(
            move_range=1,
            attack_range=0,
            attack_strength=1,
            attack_accuracy=1,
            **kwargs
        )

class MazeNaviationSim(GridWorldSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

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
        self.rewards = {agent.id: 0 for agent in self.agents.values()}
    
    def step(self, action_dict, **kwargs):
        # Process moves
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            move_result = self.move_actor.process_action(agent, action, **kwargs)
            if not move_result:
                self.rewards[agent.id] -= 0.1
        
        # Entropy penalty
        for agent_id in action_dict:
            self.rewards[agent_id] -= 0.01
        
        # Reached the target
        # ...
    
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
        agent = self.agents[agent_id]
        return {
            **self.grid_observer.get_obs(agent, **kwargs)
        }

    def get_reward(self, agent_id, **kwargs):
        reward = self.rewards[agent_id]
        self.rewards[agent_id] = 0
        return reward

    def get_done(self, agent_id, **kwargs):
        pass

    def get_all_done(self, **kwargs):
        pass

    def get_info(self, agent_id, **kwargs):
        return {}

if __name__ == "__main__":
    object_registry = {
        'N': lambda n: MazeNavigationAgent(
            id=f'navigator{n}',
            encoding=1,
            view_range=2,
            render_color='blue',
        ),
        'T': lambda n: GridWorldAgent(
            id=f'target{n}',
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
        overlapping={1: [3], 3: [1]},
        attack_mapping={1: [3]}
    )
    sim.reset()
    fig = plt.figure()
    sim.render(fig=fig)

    plt.show()
