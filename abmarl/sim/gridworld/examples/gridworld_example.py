
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.agent import GridWorldAgent, GridObservingAgent, MovingAgent, \
    AttackingAgent, HealthAgent
from abmarl.sim.gridworld.state import PositionState, HealthState
from abmarl.sim.gridworld.actor import MoveActor, AttackActor
from abmarl.sim.gridworld.observer import SingleGridObserver
from abmarl.tools.matplotlib_utils import mscatter


class WallAgent(GridWorldAgent):
    """
    Wall agents, immobile, and view blocking.
    """
    def __init__(self, **kwargs):
        super().__init__(view_blocking=True, **kwargs)


class FoodAgent(HealthAgent):
    """
    Food Agents do not move and can be attacked by Foraging Agents.
    """
    pass


class ForagingAgent(HealthAgent, AttackingAgent, MovingAgent, GridObservingAgent):
    """
    Foraging Agents can move, attack Food agents, and be attacked by Hunting agents.
    """
    pass


class HuntingAgent(HealthAgent, AttackingAgent, MovingAgent, GridObservingAgent):
    """
    Hunting agents can move and attack Foraging agents.
    """
    pass


class GridSim(GridWorldSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State Components
        self.position_state = PositionState(**kwargs)
        self.health_state = HealthState(**kwargs)

        # Action Components
        self.attack_actor = AttackActor(health_state=self.health_state, **kwargs)
        self.move_actor = MoveActor(position_state=self.position_state, **kwargs)

        # Observation Components
        self.grid_observer = SingleGridObserver(**kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.health_state.reset(**kwargs)

    def step(self, action_dict, **kwargs):
        # Process attacks:
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.active:
                self.attack_actor.process_action(agent, action, **kwargs)

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
        f'wall{i}': WallAgent(id=f'wall{i}', encoding=1, render_shape='X') for i in range(7)
    }
    food = {
        f'food{i}': FoodAgent(id=f'food{i}', encoding=2, initial_health=1, render_shape='s') for i in range(5)
    }
    foragers = {
        f'forager{i}': ForagingAgent(
            id=f'forager{i}', initial_health=1, move_range=1, attack_range=1, attack_strength=1,
            attack_accuracy=1, view_range=4, encoding=3, render_shape='o'
        ) for i in range(3)
    }
    hunters = {
        f'hunter{i}': HuntingAgent(
            id=f"hunter{i}", initial_health=1, move_range=1, attack_range=2, attack_strength=1,
            attack_accuracy=1, view_range=3, encoding=4, render_shape='D'
        ) for i in range(1)
    }
    agents = {**walls, **food, **foragers, **hunters}

    # Create simulation
    attack_mapping = {
        3: [2],
        4: [3]
    }
    sim = GridSim.build_sim(
        rows=8, cols=12, agents=agents, attack_mapping=attack_mapping
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

    # Examine the agents' observations
    from pprint import pprint
    for agent in agents.values():
        if isinstance(agent, ObservingAgent) and agent.active:
            print(agent.position)
            pprint(sim.get_obs(agent.id)['grid'])
            print()

    # plt.show()

    # Test a reset
    sim.reset()
    sim.render(fig=fig)

    # Examine the agents' observations
    from pprint import pprint
    for agent in agents.values():
        if isinstance(agent, ObservingAgent) and agent.active:
            print(agent.position)
            pprint(sim.get_obs(agent.id)['grid'])
            print()

    plt.show()
