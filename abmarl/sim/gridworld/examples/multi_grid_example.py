
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.agent import GridWorldAgent, GridObservingAgent, MovingAgent, \
    AttackingAgent, HealthAgent
from abmarl.sim.gridworld.state import PositionState, HealthState
from abmarl.sim.gridworld.actor import MoveActor, AttackActor
from abmarl.sim.gridworld.observer import MultiGridObserver


class WallAgent(GridWorldAgent):
    """
    Wall agents, immobile, and blocking.
    """
    def __init__(self, **kwargs):
        super().__init__(blocking=True, **kwargs)


class TreasureAgent(HealthAgent):
    """
    Food Agents do not move and can be attacked by Foraging Agents.
    """
    pass


class ExploringAgent(MovingAgent, GridObservingAgent):
    """
    Foraging Agents can move, attack Food agents, and be attacked by Hunting agents.
    """
    pass


class GridSim(GridWorldSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State Components
        self.position_state = PositionState(**kwargs)
        self.health_state = HealthState(**kwargs)

        # Action Components
        self.move_actor = MoveActor(position_state=self.position_state, **kwargs)

        # Observation Components
        self.grid_observer = MultiGridObserver(**kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.health_state.reset(**kwargs)

    def step(self, action_dict, **kwargs):
        # Process moves
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.active:
                self.move_actor.process_action(agent, action, **kwargs)

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
    treasure = {
        f'treasure{i}': TreasureAgent(id=f'treasure{i}', encoding=2, initial_health=1, render_shape='s') for i in range(35)
    }
    explorers = {
        f'explorer{i}': ExploringAgent(
            id=f'explorer{i}', initial_health=1, move_range=1, view_range=4, encoding=3, render_shape='o'
        ) for i in range(3)
    }
    agents = {**walls, **treasure, **explorers}

    # Create simulation
    overlap_matrix = {
        2: [3],
        3: [2, 3],
    }
    sim = GridSim.build_sim(
        rows=8, cols=12, agents=agents, overlapping=overlap_matrix
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
            obs = sim.get_obs(agent.id)['grid']
            for i in range(3):
                pprint(obs[:,:,i])
            print()

    plt.show()
