from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim.components.state import ContinuousPositionState, SpeedAngleState, LifeState
from abmarl.sim.components.actor import SpeedAngleMovementActor, AttackActor
from abmarl.sim.components.observer import SpeedObserver, AngleObserver, PositionObserver, \
    LifeObserver, HealthObserver
from abmarl.sim.components.done import DeadDone
from abmarl.sim.components.agent import SpeedAngleAgent, SpeedAngleActingAgent, AttackingAgent, \
    SpeedAngleObservingAgent, PositionObservingAgent, LifeObservingAgent, HealthObservingAgent
from abmarl.sim import AgentBasedSimulation
from abmarl.tools.matplotlib_utils import mscatter


class FightingBirdAgent(
    SpeedAngleAgent, SpeedAngleActingAgent, AttackingAgent, SpeedAngleObservingAgent,
    PositionObservingAgent, LifeObservingAgent, HealthObservingAgent
): pass


class FightingBirdsSim(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State
        self.position_state = ContinuousPositionState(**kwargs)
        self.speed_angle_state = SpeedAngleState(**kwargs)
        self.life_state = LifeState(**kwargs)

        # Actor
        self.move_actor = SpeedAngleMovementActor(
            position_state=self.position_state, speed_angle_state=self.speed_angle_state, **kwargs
        )
        self.attack_actor = AttackActor(**kwargs)

        # Observer
        self.position_observer = PositionObserver(position_state=self.position_state, **kwargs)
        self.speed_observer = SpeedObserver(**kwargs)
        self.angle_observer = AngleObserver(**kwargs)
        self.health_observer = HealthObserver(**kwargs)
        self.life_observer = LifeObserver(**kwargs)

        # Done
        self.done = DeadDone(**kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.speed_angle_state.reset(**kwargs)
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
            self.move_actor.process_move(
                self.agents[agent_id], action.get('accelerate', np.zeros(1)),
                action.get('bank', np.zeros(1)), **kwargs
            )

    def render(self, fig=None, **kwargs):
        fig.clear()
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}

        # Draw the resources
        ax = fig.gca()

        # Draw the agents
        ax.set(xlim=(0, self.position_state.region), ylim=(0, self.position_state.region))
        ax.set_xticks(np.arange(0, self.position_state.region, 1))
        ax.set_yticks(np.arange(0, self.position_state.region, 1))

        agents_x = [
            agent.position[0] for agent in self.agents.values() if render_condition[agent.id]
        ]
        agents_y = [
            agent.position[1] for agent in self.agents.values() if render_condition[agent.id]
        ]
        mscatter(agents_x, agents_y, ax=ax, m='o', s=100, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)

    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return {
            **self.position_observer.get_obs(agent, **kwargs),
            **self.speed_observer.get_obs(agent, **kwargs),
            **self.angle_observer.get_obs(agent, **kwargs),
            **self.health_observer.get_obs(agent, **kwargs),
            **self.life_observer.get_obs(agent, **kwargs),
        }

    def get_reward(self, agent_id, **kwargs):
        pass

    def get_done(self, agent_id, **kwargs):
        return self.done.get_done(self.agents[agent_id], **kwargs)

    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)

    def get_info(self, agent_id, **kwargs):
        pass


if __name__ == "__main__":
    agents = {
        f'bird{i}': FightingBirdAgent(
            id=f'bird{i}', min_speed=0.5, max_speed=1.0, max_acceleration=0.1,
            max_banking_angle=90, max_banking_angle_change=90,
            initial_banking_angle=45, attack_range=1.0, attack_strength=0.5
        ) for i in range(24)
    }

    sim = FightingBirdsSim(
        region=20,
        agents=agents,
        attack_norm=2,
    )
    fig = plt.figure()
    sim.reset()
    sim.render(fig=fig)

    print(sim.get_obs('bird0'))

    for i in range(50):
        action_dict = {
            agent.id: agent.action_space.sample() for agent in sim.agents.values()
            if agent.is_alive
        }
        sim.step(action_dict)
        sim.render(fig=fig)
        for agent in agents:
            print(agent, ': ', sim.get_done(agent))
        print('\n')

    print(sim.get_all_done())
