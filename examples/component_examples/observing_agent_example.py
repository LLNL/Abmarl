
from matplotlib import pyplot as plt
import numpy as np

from admiral.envs.components.team import TeamAgent
from admiral.envs.components.observer import ObservingAgent
from admiral.envs.components.position import PositionAgent, PositionState, GridPositionTeamBasedObserver
from admiral.envs.components.movement import GridMovementActor, GridMovementAgent
from admiral.envs import AgentBasedSimulation
from admiral.tools.matplotlib_utils import mscatter

class ObservingTeamMovementAgent(ObservingAgent, TeamAgent, PositionAgent, GridMovementAgent):
    pass

class SimpleGridObservations(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State components
        self.position_state = PositionState(**kwargs)

        # Actor components
        self.move_actor = GridMovementActor(position=self.position_state, **kwargs)

        # Observers
        self.observer = GridPositionTeamBasedObserver(position=self.position_state, **kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)

        return {'agent0': self.observer.get_obs(self.agents['agent0'])}
    
    def step(self, action_dict, **kwargs):

        # Process movement
        for agent_id, action in action_dict.items():
            self.move_actor.process_move(self.agents[agent_id], action['move'], **kwargs)

        return {'agent0': self.observer.get_obs(self.agents['agent0'])}
    
    def render(self, fig=None, **kwargs):
        fig.clear()

        # Draw the agents
        team_shapes = {
            0: 'o',
            1: 's',
            2: 'd'
        }

        ax = fig.gca()
        shape_dict = {agent.id: team_shapes[agent.team] for agent in self.agents.values()}
        ax.set(xlim=(0, self.position_state.region), ylim=(0, self.position_state.region))
        ax.set_xticks(np.arange(0, self.position_state.region, 1))
        ax.set_yticks(np.arange(0, self.position_state.region, 1))
        ax.grid()

        agents_x = [agent.position[1] + 0.5 for agent in self.agents.values()]
        agents_y = [self.position_state.region - 0.5 - agent.position[0] for agent in self.agents.values()]
        shape = [shape_dict[agent_id] for agent_id in shape_dict]
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        return self.position.get_obs(agent_id, **kwargs)
    
    def get_reward(self, agent_id, **kwargs):
        return self.rewarder.get_reward(agent_id, **kwargs)
    
    def get_done(self, agent_id, **kwargs):
        pass
    
    def get_all_done(self, agent_id, **kwargs):
        pass
    
    def get_info(self, agent_id, **kwargs):
        return {}


agents = {
    'agent0': ObservingTeamMovementAgent(id='agent0', team=0, view=1, move_range=1, starting_position=np.array([2, 1])),
    'agent1': ObservingTeamMovementAgent(id='agent1', team=0, view=1, move_range=0, starting_position=np.array([2, 2])),
    'agent2': ObservingTeamMovementAgent(id='agent2', team=1, view=1, move_range=0, starting_position=np.array([0, 4])),
    'agent3': ObservingTeamMovementAgent(id='agent3', team=1, view=1, move_range=0, starting_position=np.array([0, 0])),
    'agent4': ObservingTeamMovementAgent(id='agent4', team=2, view=1, move_range=0, starting_position=np.array([4, 0])),
    'agent5': ObservingTeamMovementAgent(id='agent5', team=2, view=1, move_range=0, starting_position=np.array([4, 4])),
}
env = SimpleGridObservations(
    region=5,
    agents=agents,
    number_of_teams=3
)
obs = env.reset()
fig = plt.gcf()
env.render(fig=fig)
print(obs['agent0'])
print(obs['agent0'])
print(obs['agent0'])
print()

obs = env.step({'agent0': {'move': np.array([-1, 0])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([0, 1])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([0, 1])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([1, 0])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([1, 0])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([0, -1])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([0, -1])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([-1, 0])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

plt.show()
