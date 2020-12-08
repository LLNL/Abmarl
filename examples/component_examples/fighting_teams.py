
from matplotlib import pyplot as plt

from admiral.component_envs.observer import ObservingAgent
from admiral.component_envs.team import TeamAgent
from admiral.component_envs.position import GridPositionTeamsComponent, GridPositionAgent
from admiral.component_envs.movement import GridMovementComponent, GridMovementAgent
from admiral.component_envs.attacking import GridAttackingTeamComponent, GridAttackingAgent
from admiral.component_envs.death_life import DyingComponent, LifeAgent
from admiral.component_envs.rewarder import RewarderComponent
from admiral.component_envs.done_component import TeamDeadDoneComponent
from admiral.envs import AgentBasedSimulation

class FightingTeamsAgent(LifeAgent, GridPositionAgent, GridAttackingAgent, TeamAgent, GridMovementAgent, ObservingAgent):
    pass

class FightingTeamsEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.position = GridPositionTeamsComponent(**kwargs)
        self.movement = GridMovementComponent(**kwargs)
        self.attacking = GridAttackingTeamComponent(**kwargs)
        self.dying = DyingComponent(**kwargs)
        self.rewarder = RewarderComponent(**kwargs)
        self.done_conditioner = TeamDeadDoneComponent(**kwargs)

        self.finalize()

        self.attacking_record = []
    
    def reset(self, **kwargs):
        self.position.reset(**kwargs)
        self.dying.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.is_alive:
                if action.get('attack', False):
                    attacked_agent = self.attacking.process_attack(agent)
                    if attacked_agent is not None:
                        self.agents[attacked_agent].add_health(-agent.attack_strength)
                        self.attacking_record.append(agent.id + " attacked " + attacked_agent)
                if 'move' in action:
                    self.movement.process_move(agent, action['move'])
    
    def render(self, fig=None, **kwargs):
        fig.clear()
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        shape = {agent.id: 'o' if agent.team == 1 else 's' for agent in self.agents.values()}
        self.position.render(fig=fig, render_condition=render_condition, shape_dict=shape, **kwargs)
        for record in self.attacking_record:
            print(record)
        self.attacking_record.clear()
        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        return {'agents': self.position.get_obs(agent_id)}
    
    def get_reward(self, agent_id, **kwargs):
        self.rewarder.get_reward(agent_id)

    def get_done(self, agent_id, **kwargs):
        return self.done_conditioner.get_done(agent_id)
    
    def get_all_done(self, **kwargs):
        return self.done_conditioner.get_all_done(**kwargs)
    
    def get_info(self, **kwargs):
        return {}

agents = {f'agent{i}': FightingTeamsAgent(
    id=f'agent{i}', attack_range=1, attack_strength=0.4, team=i%2, move=1, view=11
) for i in range(24)}
env = FightingTeamsEnv(
    region=12,
    agents=agents,
    number_of_teams=2
)
env.reset()
print({agent_id: env.get_obs(agent_id) for agent_id in env.agents})
fig = plt.gcf()
env.render(fig=fig)

for _ in range(100):
    action_dict = {}
    for agent_id, agent in env.agents.items():
        action_dict[agent_id] = agent.action_space.sample()
    env.step(action_dict)
    env.render(fig=fig)
    print(env.get_all_done())
    x = []

