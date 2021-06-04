from matplotlib import pyplot as plt
import numpy as np

from admiral.envs.components.agent import \
    AttackingAgent, BroadcastingAgent, GridMovementAgent, \
    PositionObservingAgent, LifeObservingAgent, TeamObservingAgent, AgentObservingAgent
from admiral.envs.components.state import GridPositionState, BroadcastState, LifeState
from admiral.envs.components.actor import GridMovementActor, AttackActor, BroadcastActor
from admiral.envs.components.observer import PositionObserver, LifeObserver, TeamObserver
from admiral.envs.components.done import TeamDeadDone
from admiral.envs.components.wrappers.observer_wrapper import \
    PositionRestrictedObservationWrapper, TeamBasedCommunicationWrapper
from admiral.envs import AgentBasedSimulation
from admiral.tools.matplotlib_utils import mscatter


class AllChannelsObservingAgent(
    PositionObservingAgent, LifeObservingAgent, TeamObservingAgent, AgentObservingAgent
): pass
class CommunicatingAgent(BroadcastingAgent, AllChannelsObservingAgent): pass
class BattleAgent(AttackingAgent, GridMovementAgent, AllChannelsObservingAgent): pass


class TeamBattleCommsEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # state
        self.position_state = GridPositionState(**kwargs)
        self.life_state = LifeState(**kwargs)
        self.broadcast_state = BroadcastState(**kwargs)

        # observer
        position_observer = PositionObserver(position_state=self.position_state, **kwargs)
        life_observer = LifeObserver(**kwargs)
        team_observer = TeamObserver(**kwargs)
        partial_observer = PositionRestrictedObservationWrapper(
            [position_observer, team_observer, life_observer], **kwargs
        )
        self.comms_observer = TeamBasedCommunicationWrapper([partial_observer], **kwargs)

        # actor
        self.move_actor = GridMovementActor(position_state=self.position_state, **kwargs)
        self.attack_actor = AttackActor(**kwargs)
        self.broadcast_actor = BroadcastActor(broadcast_state=self.broadcast_state, **kwargs)

        # done
        self.done = TeamDeadDone(**kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.life_state.reset(**kwargs)
        self.broadcast_state.reset(**kwargs)

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

        # Process broadcasting
        for agent_id, action in action_dict.items():
            self.broadcast_actor.process_action(self.agents[agent_id], action, **kwargs)

    def render(self, fig=None, **kwargs):
        fig.clear()
        ax = fig.gca()

        # Draw the agents
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        shape_dict = {agent.id: 'o' if agent.team == 1 else 's' for agent in self.agents.values()}
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
        shape = [shape_dict[agent_id] for agent_id in shape_dict if render_condition[agent_id]]
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)

    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return self.comms_observer.get_obs(agent, **kwargs)

    def get_reward(self, agent_id, **kwargs):
        pass

    def get_done(self, agent_id, **kwargs):
        return self.done.get_done(self.agents[agent_id], **kwargs)

    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)

    def get_info(self, agent_id, **kwargs):
        return {}


if __name__ == "__main__":
    agents = {
        'agent0': CommunicatingAgent(
            id='agent0', initial_position=np.array([7, 7]), team=1, broadcast_range=11,
            agent_view=11
        ),
        'agent1': BattleAgent(
            id='agent1', initial_position=np.array([0, 4]), team=1, agent_view=2, attack_range=1,
            move_range=1, attack_strength=1
        ),
        'agent2': BattleAgent(
            id='agent2', initial_position=np.array([0, 7]), team=1, agent_view=2, attack_range=1,
            move_range=1, attack_strength=1
        ),
        'agent3': BattleAgent(
            id='agent3', initial_position=np.array([0, 10]), team=1, agent_view=2, attack_range=1,
            move_range=1, attack_strength=1
        ),
        'agent4': BattleAgent(
            id='agent4', initial_position=np.array([14, 4]), team=2, agent_view=2, attack_range=1,
            move_range=1, attack_strength=1
        ),
        'agent5': BattleAgent(
            id='agent5', initial_position=np.array([14, 7]), team=2, agent_view=2, attack_range=1,
            move_range=1, attack_strength=1
        ),
        'agent6': BattleAgent(
            id='agent6', initial_position=np.array([14, 10]), team=2, agent_view=2, attack_range=1,
            move_range=1, attack_strength=1
        ),
    }
    env = TeamBattleCommsEnv(
        region=15,
        agents=agents,
        number_of_teams=2
    )
    env.reset()
    fig = plt.figure()
    env.render(fig=fig)

    for _ in range(50):
        action_dict = {
            agent.id: agent.action_space.sample() for agent in env.agents.values()
            if agent.is_alive
        }
        env.step(action_dict)
        env.render(fig=fig)
        print(env.get_all_done())
