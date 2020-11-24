
from matplotlib import pyplot as plt
import numpy as np

from admiral.component_envs.world import GridWorldTeamsComponent, GridWorldObservingTeamAgent
from admiral.component_envs.movement import GridWorldMovementComponent, GridWorldMovementAgent
from admiral.component_envs.attacking import GridAttackingTeamComponent, GridWorldAttackingTeamAgent
from admiral.component_envs.death_life import DyingComponent, DyingAgent
from admiral.component_envs.resources import GridResourceComponent, GridResourceHarvestingAndObservingAgent, GridResourceObservingAgent
from admiral.component_envs.rewarder import RewarderComponent
from admiral.component_envs.done_conditioner import TeamDeadDoneComponent
from admiral.envs import AgentBasedSimulation

class PreyAgent(GridWorldObservingTeamAgent, GridWorldMovementAgent, DyingAgent, GridResourceHarvestingAndObservingAgent):
    pass

class PredatorAgent(GridWorldObservingTeamAgent, GridWorldMovementAgent, GridWorldAttackingTeamAgent, DyingAgent, GridResourceObservingAgent):
    pass

class PredatorPreyEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.world = GridWorldTeamsComponent(**kwargs)
        self.movement = GridWorldMovementComponent(**kwargs)
        self.attacking = GridAttackingTeamComponent(**kwargs)
        self.dying = DyingComponent(**kwargs)
        self.resource = GridResourceComponent(**kwargs)
        self.rewarder = RewarderComponent(**kwargs)
        self.done_conditioner = TeamDeadDoneComponent(**kwargs)
        
        self.finalize()

        self.attacking_record = []
    
    def reset(self, **kwargs):
        self.world.reset(**kwargs)
        self.dying.reset(**kwargs)
        self.resource.reset(**kwargs)

    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.is_alive:
                if 'move' in action:
                    self.movement.act(agent, action['move'])
                if action.get('attack', False):
                    attacked_agent = self.attacking.act(agent)
                    if attacked_agent is not None:
                        self.agents[attacked_agent].health -= agent.attack_strength
                        agent.health += agent.attack_strength # Gain health from a good attack.
                        self.attacking_record.append(agent.id + " attacked " + attacked_agent)
                if action.get('harvest', False):
                    amount_harvested = self.resource.act(agent, action['harvest'])
                    agent.health += amount_harvested

        for agent_id in action_dict:
            agent = self.agents[agent_id]
            if agent.is_alive:
                self.dying.apply_entropy(agent)
                self.dying.process_death(agent)
        # TODO: There is likely a bug in the entropy or attack health because prey are getting attacked
        # many more times then they should be alive...

        self.resource.regrow()
    
    def render(self, fig=None, **kwargs):
        fig.clear()
        self.resource.render(fig=fig, **kwargs)
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        shape = {agent.id: 'o' if agent.team == 1 else 's' for agent in self.agents.values()}
        self.world.render(fig=fig, render_condition=render_condition, shape_dict=shape, **kwargs)
        for record in self.attacking_record:
            print(record)
        self.attacking_record.clear()
        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
            return {'agents': self.world.get_obs(agent_id), 'resources': self.resource.get_obs(agent_id)}
    
    def get_reward(self, agent_id, **kwargs):
        self.rewarder.get_reward(agent_id)

    def get_done(self, agent_id, **kwargs):
        return self.done_conditioner.get_done(agent_id)
    
    def get_all_done(self, **kwargs):
        return self.done_conditioner.get_all_done(**kwargs)
    
    def get_info(self, **kwargs):
        return {}

prey = {f'prey{i}': PreyAgent(id=f'prey{i}', view=5, team=1, move=1, attack_range=-1, attack_strength=0.0, max_harvest=0.5) for i in range(7)}
predators = {f'predator{i}': PredatorAgent(id=f'predator{i}', view=2, team=2, move=1, attack_range=1, attack_strength=0.24, max_harvest=0.0) for i in range(2)}
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

