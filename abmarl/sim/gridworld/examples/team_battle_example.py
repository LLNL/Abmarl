
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.agent import GridObservingAgent, MovingAgent, AttackingAgent, HealthAgent
from abmarl.sim.gridworld.state import HealthState, PositionState
from abmarl.sim.gridworld.actor import MoveActor, AttackActor
from abmarl.sim.gridworld.observer import SingleGridObserver
from abmarl.sim.gridworld.done import OneTeamRemainingDone


class BattleAgent(GridObservingAgent, MovingAgent, AttackingAgent, HealthAgent):
    def __init__(self, **kwargs):
        super().__init__(
            move_range=1,
            attack_range=1,
            attack_strength=1,
            attack_accuracy=1,
            view_range=3,
            **kwargs
        )


class TeamBattleSim(GridWorldSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State Components
        self.position_state = PositionState(**kwargs)
        self.health_state = HealthState(**kwargs)

        # Action Components
        self.move_actor = MoveActor(**kwargs)
        self.attack_actor = AttackActor(**kwargs)

        # Observation Components
        self.grid_observer = SingleGridObserver(**kwargs)

        # Done Compoennts
        self.done = OneTeamRemainingDone(**kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.health_state.reset(**kwargs)

        # Track the rewards
        self.rewards = {agent.id: 0 for agent in self.agents.values()}

    def step(self, action_dict, **kwargs):
        # Process attacks:
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            attacked_agent = self.attack_actor.process_action(agent, action, **kwargs)
            if attacked_agent is not None:
                self.rewards[attacked_agent.id] -= 1
                self.rewards[agent.id] += 1
            else:
                self.rewards[agent.id] -= 0.1

        # Process moves
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.active:
                move_result = self.move_actor.process_action(agent, action, **kwargs)
                if not move_result:
                    self.rewards[agent.id] -= 0.1

        # Entropy penalty
        for agent_id in action_dict:
            self.rewards[agent_id] -= 0.01

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
        return self.done.get_done(self.agents[agent_id])

    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)

    def get_info(self, agent_id, **kwargs):
        return {}


if __name__ == "__main__":
    colors = ['red', 'blue', 'green', 'gray']
    positions = [np.array([1, 1]), np.array([1, 6]), np.array([6, 1]), np.array([6, 6])]
    agents = {
        f'agent{i}': BattleAgent(
            id=f'agent{i}',
            encoding=i % 4 + 1,
            render_color=colors[i % 4],
            initial_position=positions[i % 4]
        ) for i in range(24)
    }
    overlap_map = {
        1: [1],
        2: [2],
        3: [3],
        4: [4]
    }
    attack_map = {
        1: [2, 3, 4],
        2: [1, 3, 4],
        3: [1, 2, 4],
        4: [1, 2, 3]
    }
    sim = TeamBattleSim.build_sim(
        8, 8,
        agents=agents,
        overlapping=overlap_map,
        attack_mapping=attack_map
    )

    sim.reset()
    fig = plt.figure()
    sim.render(fig=fig)

    done_agents = set()
    for i in range(50):
        action = {
            agent.id: agent.action_space.sample()
            for agent in agents.values() if agent.id not in done_agents
        }
        sim.step(action)
        sim.render(fig=fig)

        if sim.get_all_done():
            break
        for agent in agents:
            if sim.get_done(agent):
                done_agents.add(agent)
