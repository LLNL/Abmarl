
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.agent import GridWorldAgent, GridObservingAgent, MovingAgent, \
    AttackingAgent, HealthAgent
from abmarl.sim.gridworld.state import OverlappingPositionState, HealthState
from abmarl.sim.gridworld.actor import MoveActor, AttackActor
from abmarl.sim.gridworld.observer import GridObserver
from abmarl.tools.matplotlib_utils import mscatter


class WallAgent(GridWorldAgent):
    """
    Wall agents, immobile and view blocking.

    Args:
        encoding: Default encoding is 1.
        render_shape: Default render_shape set to 'X'.
    """
    def __init__(self, encoding=1, render_shape='X', **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.render_shape = render_shape


class FoodAgent(HealthAgent):
    """
    Food Agents do not move and can be attacked by Foraging Agents.
    
    Args:
        encoding: Default encoding set to 2.
    """
    def __init__(self, encoding=2, **kwargs):
        kwargs['overlappable'] = True
        super().__init__(**kwargs)
        self.encoding = encoding


class ForagingAgent(HealthAgent, AttackingAgent, MovingAgent, GridObservingAgent):
    """
    Foraging Agents can move, attack Food agents, and be attacked by Hunting agents.
    
    Args:
        encoding: Default encoding set to 3.
        render_shape: Default render_shape set to 'o'.
    """
    def __init__(self, encoding=3, render_shape='o', **kwargs):
        kwargs['overlappable'] = True
        super().__init__(**kwargs)
        self.encoding = encoding
        self.render_shape = render_shape


class HuntingAgent(HealthAgent, AttackingAgent, MovingAgent, GridObservingAgent):
    """
    Hunting agents can move and attack Foraging agents.

    Args:
        encoding: Default encoding set to 4.
        render_shape: Default render_shape set to 'D'.
    """
    def __init__(self, encoding=4, render_shape='D', **kwargs):
        kwargs['overlappable'] = True
        super().__init__(**kwargs)
        self.encoding = encoding
        self.render_shape = render_shape


class GridSim(GridWorldSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State Components
        self.position_state = OverlappingPositionState(**kwargs)
        self.health_state = HealthState(**kwargs)

        # Action Components
        # self.attack_actor = AttackActor(health_state=self.health_state, **kwargs)
        self.move_actor = MoveActor(**kwargs)

        # Observation Components
        self.grid_observer = GridObserver(**kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.health_state.reset(**kwargs)

    def step(self, action_dict, **kwargs):
        # Process attacks:
        # for agent_id, action in action_dict.items():
        #     agent = self.agents[agent_id]
        #     if agent.active:
        #         self.attack_actor.process_action(agent, action, **kwargs)

        # Process moves
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.active:
                self.move_actor.process_action(agent, action, **kwargs)

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
    from abmarl.sim import ActingAgent, ObservingAgent

    fig = plt.figure()

    # Create agents
    walls = {
        f'wall{i}': WallAgent(id=f'wall{i}', view_blocking=True) for i in range(7)
    }
    food = {
        f'food{i}': FoodAgent(id=f'food{i}', initial_health=1) for i in range(15)
    }
    foragers = {
        f'forager{i}': ForagingAgent(
            id=f'forager{i}', initial_health=1, move_range=1, attack_range=1, attack_strength=1,
            attack_accuracy=1, view_range=4
        ) for i in range(7)
    }
    hunters = {
        f'hunter{i}': HuntingAgent(
            id=f"hunter{i}", initial_health=1, move_range=1, attack_range=2, attack_strength=1,
            attack_accuracy=1, view_range=3
        ) for i in range(2)
    }
    agents = {**walls, **food, **foragers, **hunters}

    # Create simulation
    attack_mapping = {
        3: [2],
        4: [3]
    }
    sim = GridSim.build_sim(
        rows=8, cols=12, agents=agents, attack_mapping=attack_mapping, overlapping=True
    )
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
