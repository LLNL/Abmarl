from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim.components.state import GridPositionState, LifeState
from abmarl.sim.components.observer import TeamObserver, PositionObserver, HealthObserver, \
    LifeObserver
from abmarl.sim.components.actor import GridMovementActor, AttackActor
from abmarl.sim.components.done import TeamDeadDone
from abmarl.sim.components.agent import TeamObservingAgent, PositionObservingAgent, \
    HealthObservingAgent, LifeObservingAgent, GridMovementAgent, AttackingAgent
from abmarl.sim import AgentBasedSimulation
from abmarl.tools.matplotlib_utils import mscatter


class FightingTeamsAgent(
    TeamObservingAgent, PositionObservingAgent, HealthObservingAgent,
    LifeObservingAgent, GridMovementAgent, AttackingAgent
): pass


class FightingTeamsSim(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State Components
        self.position_state = GridPositionState(**kwargs)
        self.life_state = LifeState(**kwargs)

        # Observer Components
        self.position_observer = PositionObserver(position_state=self.position_state, **kwargs)
        self.health_observer = HealthObserver(**kwargs)
        self.life_observer = LifeObserver(**kwargs)
        self.team_observer = TeamObserver(**kwargs)

        # Actor Components
        self.move_actor = GridMovementActor(position_state=self.position_state, **kwargs)
        self.attack_actor = AttackActor(**kwargs)

        # Done components
        self.done = TeamDeadDone(**kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.life_state.reset(**kwargs)

    def step(self, action_dict, **kwargs):
        # Process attacking
        for agent_id, action in action_dict.items():
            attacking_agent = self.agents[agent_id]
            attacked_agent = self.attack_actor.process_action(attacking_agent, action, **kwargs)
            if attacked_agent is not None:
                self.life_state.modify_health(attacked_agent, -attacking_agent.attack_strength)

        # Process movement
        for agent_id, action in action_dict.items():
            self.move_actor.process_action(self.agents[agent_id], action, **kwargs)

    def render(self, fig=None, **kwargs):
        fig.clear()
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        shape_dict = {agent.id: 'o' if agent.team == 1 else 's' for agent in self.agents.values()}

        ax = fig.gca()
        ax.set(xlim=(0, self.position_state.region), ylim=(0, self.position_state.region))
        ax.set_xticks(np.arange(0, self.position_state.region, 1))
        ax.set_yticks(np.arange(0, self.position_state.region, 1))
        ax.grid()

        agents_x = [
            agent.position[1] + 0.5 for agent in self.agents.values() if render_condition[agent.id]
        ]
        agents_y = [
            self.position_state.region - 0.5 - agent.position[0] for agent in self.agents.values()
            if render_condition[agent.id]
        ]

        if shape_dict:
            shape = [shape_dict[agent_id] for agent_id in shape_dict if render_condition[agent_id]]
        else:
            shape = 'o'
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)

    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return {
            **self.position_observer.get_obs(agent, **kwargs),
            **self.health_observer.get_obs(agent, **kwargs),
            **self.life_observer.get_obs(agent, **kwargs),
            **self.team_observer.get_obs(agent, **kwargs),
        }

    def get_reward(self, agent_id, **kwargs):
        pass

    def get_done(self, agent_id, **kwargs):
        return self.done.get_done(agent_id)

    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)

    def get_info(self, **kwargs):
        return {}


if __name__ == '__main__':
    agents = {f'agent{i}': FightingTeamsAgent(
        id=f'agent{i}', attack_range=1, attack_strength=0.4, team=i % 2 + 1, move_range=1
    ) for i in range(24)}
    sim = FightingTeamsSim(
        region=12,
        agents=agents,
        number_of_teams=2
    )
    sim.reset()
    print({agent_id: sim.get_obs(agent_id) for agent_id in sim.agents})
    fig = plt.gcf()
    sim.render(fig=fig)

    for _ in range(100):
        action_dict = {
            agent.id: agent.action_space.sample() for agent in sim.agents.values()
            if agent.is_alive
        }
        sim.step(action_dict)
        sim.render(fig=fig)
        print(sim.get_all_done())
