
from matplotlib import pyplot as plt
import numpy as np

from admiral.envs.components.agent import \
    PositionAgent, LifeAgent, TeamAgent, \
    AttackingAgent, BroadcastingAgent, \
    PositionObservingAgent, LifeObservingAgent, TeamObservingAgent, AgentObservingAgent
from admiral.envs.components.state import GridPositionState, BroadcastState, LifeState, TeamState
from admiral.envs.components.actor import GridMovementActor, PositionTeamBasedAttackActor, BroadcastActor
from admiral.envs.components.observer import PositionObserver, LifeObserver, TeamObserver
from admiral.envs.components.wrappers.observer_wrapper import PositionRestrictedObservationWrapper, TeamBasedCommunicationWrapper
from admiral.envs import AgentBasedSimulation
from admiral.tools.matplotlib_utils import mscatter

class CommunicatingAgent(PositionAgent, TeamAgent, BroadcastingAgent, PositionObservingAgent, TeamObservingAgent, AgentObservingAgent): pass

class BroadcastCommunicationEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        
        # state
        self.position_state = GridPositionState(**kwargs)
        # self.life_state = LifeState(**kwargs)
        self.team_state = TeamState(**kwargs)
        self.broadcast_state = BroadcastState(**kwargs)
        
        # observer
        position_observer = PositionObserver(position=self.position_state, **kwargs)
        # life_observer = LifeObserver(**kwargs)
        team_observer = TeamObserver(team=self.team_state, **kwargs)
        partial_observer = PositionRestrictedObservationWrapper([position_observer, team_observer], **kwargs)
        self.comms_observer = TeamBasedCommunicationWrapper([partial_observer], **kwargs)

        # actor
        # self.move_actor = GridMovementActor(position=self.position_state, **kwargs)
        # self.attack_actor = PositionTeamBasedAttackActor(**kwargs)
        self.broadcast_actor = BroadcastActor(broadcast_state=self.broadcast_state, **kwargs)
    
        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.broadcast_state.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        # Process broadcasting at the end
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            self.broadcast_actor.process_broadcast(agent, action.get('broadcast', 0), **kwargs)
    
    def render(self, fig=None, **kwargs):
        fig.clear()

        # Draw the agents
        team_shapes = {
            0: 'o',
            1: 's',
            2: 'd'
        }

        ax = fig.gca()
        ax.set(xlim=(0, self.position_state.region), ylim=(0, self.position_state.region))
        ax.set_xticks(np.arange(0, self.position_state.region, 1))
        ax.set_yticks(np.arange(0, self.position_state.region, 1))
        ax.grid()

        agents_x = [agent.position[1] + 0.5 for agent in self.agents.values()]
        agents_y = [self.position_state.region - 0.5 - agent.position[0] for agent in self.agents.values()]
        mscatter(agents_x, agents_y, ax=ax, m='o', s=200, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return self.comms_observer.get_obs(agent, **kwargs)
    
    def get_reward(self, agent_id, **kwargs):
        pass
    
    def get_done(self, agent_id, **kwargs):
        pass
    
    def get_all_done(self, **kwargs):
        pass
    
    def get_info(self, agent_id, **kwargs):
        return {}

if __name__ == "__main__":
    agents = {
        'agent0': CommunicatingAgent(id='agent0', initial_position=np.array([1, 7]), team=0, broadcast_range=0, agent_view=0),
        'agent1': CommunicatingAgent(id='agent1', initial_position=np.array([3, 3]), team=0, broadcast_range=4, agent_view=3),
        'agent2': CommunicatingAgent(id='agent2', initial_position=np.array([5, 0]), team=1, broadcast_range=4, agent_view=2),
        'agent3': CommunicatingAgent(id='agent3', initial_position=np.array([6, 9]), team=1, broadcast_range=4, agent_view=2),
        'agent4': CommunicatingAgent(id='agent4', initial_position=np.array([4, 7]), team=1, broadcast_range=4, agent_view=3),
    }
    env = BroadcastCommunicationEnv(
        region=10,
        agents=agents,
        number_of_teams=2
    )
    env.reset()
    fig = plt.figure()
    env.render(fig=fig)

    action_dict = {agent.id: agent.action_space.sample() for agent in env.agents.values()}
    # action_dict = {agent.id: {'broadcast': 0} for agent in env.agents.values()}
    print('\nActions:')
    print(action_dict)
    env.step(action_dict)
    print('Observations:')
    print(env.get_obs('agent0'))
    # env.render(fig=fig)

