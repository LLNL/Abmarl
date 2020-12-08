
from matplotlib import pyplot as plt

from admiral.envs.components.observer import ObservingAgent
from admiral.envs.components.team import TeamAgent
from admiral.envs.components.position import GridPositionTeamsComponent, GridPositionAgent
from admiral.envs.components.movement import GridMovementComponent, GridMovementAgent
from admiral.envs.components.attacking import GridAttackingTeamComponent, GridAttackingAgent
from admiral.envs.components.death_life import DyingComponent, LifeAgent
from admiral.envs.components.resources import GridResourceComponent, GridResourceHarvestingAgent
from admiral.envs.components.rewarder import RewarderComponent
from admiral.envs.components.done_component import TeamDeadDoneComponent
from admiral.envs import AgentBasedSimulation

class PreyAgent(GridPositionAgent, ObservingAgent, TeamAgent, GridMovementAgent, LifeAgent, GridResourceHarvestingAgent):
    pass

class PredatorAgent(GridPositionAgent, ObservingAgent, TeamAgent, GridMovementAgent, GridAttackingAgent, LifeAgent):
    pass

class PredatorPreyEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.position = GridPositionTeamsComponent(**kwargs)
        self.movement = GridMovementComponent(**kwargs)
        self.attacking = GridAttackingTeamComponent(**kwargs)
        self.dying = DyingComponent(**kwargs)
        self.resource = GridResourceComponent(**kwargs)
        self.rewarder = RewarderComponent(**kwargs)
        self.done_conditioner = TeamDeadDoneComponent(**kwargs)
        
        self.finalize()

        self.attacking_record = []
    
    def reset(self, **kwargs):
        self.position.reset(**kwargs)
        self.dying.reset(**kwargs)
        self.resource.reset(**kwargs)

    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.is_alive:
                if 'move' in action:
                    self.movement.process_move(agent, action['move'])
                if action.get('attack', False):
                    attacked_agent = self.attacking.process_attack(agent)
                    if attacked_agent is not None:
                        self.agents[attacked_agent].add_health(-agent.attack_strength)
                        agent.add_health(agent.attack_strength) # Gain health from a good attack.
                        self.attacking_record.append(agent.id + " attacked " + attacked_agent)
                if action.get('harvest', False):
                    amount_harvested = self.resource.process_harvest(agent, action['harvest'])
                    agent.add_health(amount_harvested)

        for agent_id in action_dict:
            agent = self.agents[agent_id]
            if agent.is_alive:
                self.dying.apply_entropy(agent)

        self.resource.regrow()
    
    def render(self, fig=None, **kwargs):
        fig.clear()
        self.resource.render(fig=fig, **kwargs)
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        shape = {agent.id: 'o' if agent.team == 1 else 's' for agent in self.agents.values()}
        self.position.render(fig=fig, render_condition=render_condition, shape_dict=shape, **kwargs)
        for record in self.attacking_record:
            print(record)
        self.attacking_record.clear()
        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
            return {'agents': self.position.get_obs(agent_id), 'resources': self.resource.get_obs(agent_id)}
    
    def get_reward(self, agent_id, **kwargs):
        self.rewarder.get_reward(agent_id)

    def get_done(self, agent_id, **kwargs):
        return self.done_conditioner.get_done(agent_id)
    
    def get_all_done(self, **kwargs):
        return self.done_conditioner.get_all_done(**kwargs)
    
    def get_info(self, **kwargs):
        return {}

prey = {f'prey{i}': PreyAgent(id=f'prey{i}', view=5, team=0, move=1, attack_range=-1, attack_strength=0.0, max_harvest=0.5) for i in range(7)}
predators = {f'predator{i}': PredatorAgent(id=f'predator{i}', view=2, team=1, move=1, attack_range=1, attack_strength=0.24, max_harvest=0.0) for i in range(2)}
agents = {**prey, **predators}
region = 10
env = PredatorPreyEnv(
    region=region,
    agents=agents,
    number_of_teams=2,
    entropy=0.05
)
env.reset()
print({agent_id: env.get_obs(agent_id) for agent_id in env.agents})
fig = plt.gcf()
env.render(fig=fig)

for _ in range(50):
    action_dict = {agent.id: agent.action_space.sample() for agent in env.agents.values()}
    env.step(action_dict)
    env.render(fig=fig)
    print(env.get_all_done())
    x = []

