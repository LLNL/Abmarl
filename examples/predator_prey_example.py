
from matplotlib import pyplot as plt
import numpy as np

from admiral.component_envs.world import GridWorldTeamsEnv, GridWorldTeamAgent
from admiral.component_envs.movement import GridMovementEnv, GridMovementAgent
from admiral.component_envs.attacking import GridAttackingTeamEnv, GridAttackingTeamAgent
from admiral.component_envs.death_life import DyingEnv, DyingAgent
from admiral.component_envs.resources import GridResourceEnv, GridResourceAgent

class PreyAgent(GridWorldTeamAgent, GridMovementAgent, DyingAgent, GridResourceAgent):
    pass

class PredatorAgent(GridMovementAgent, GridAttackingTeamAgent, DyingAgent):
    pass
# TODO: Resolve the Method resolution bug between world and attacking agents

class PredatorPreyEnv:
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.world = GridWorldTeamsEnv(**kwargs)
        self.movement = GridMovementEnv(**kwargs)
        self.attacking = GridAttackingTeamEnv(**kwargs)
        self.dying = DyingEnv(**kwargs)
        self.resource = GridResourceEnv(**kwargs)

        # This is good code to have after the observation and action space have been built by the
        # modules, we put them all together into a Dict.
        from gym.spaces import Dict
        for agent in self.agents.values():
            agent.action_space = Dict(agent.action_space)
            agent.observation_space = Dict(agent.observation_space)

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
                    agent.position = self.movement.process_move(agent.position, action['move'])
                if action.get('attack', False):
                    attacked_agent = self.attacking.process_attack(agent)
                    if attacked_agent is not None:
                        self.agents[attacked_agent].health -= agent.attack_strength
                        agent.health += agent.attack_strength # Gain health from a good attack.
                        self.attacking_record.append(agent.id + " attacked " + attacked_agent)
                if action.get('harvest', False):
                    amount_harvested = self.resource.process_harvest(tuple(agent.position), action['harvest'])
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
        # The observation that is returned actually depends on the agent type. Predators will just
        # see agents, so only world will return. Prey will see agents and resources, so both will
        # return.
        # TODO: The bigger probem there is that I want the predators to observe
        # the resources as well, I just don't want them to have the harvest action
        # in their action space. This means that the resources module needs to
        # know about the predators in its self.agents in order to give them the
        # correct observation_space and to get the correct get_obs, but it must
        # not consider it for the processing action.
        # Hmm.... What if resources.get_obs didn't
        # take the agent's id but just took the values it needed to know, namely
        # position and view? That's fine for get obs, but it doesn't solve the
        # problem because the agent's observation_space still needs the resources.
        if isinstance(self.agents[agent_id], PredatorAgent):
            return {'agents': self.world.get_obs(agent_id)}
        elif isinstance(self.agents[agent_id], PreyAgent):
            return {'agents': self.world.get_obs(agent_id), 'resources': self.resource.get_obs(agent_id)}

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

for _ in range(30):
    action_dict = {agent.id: agent.action_space.sample() for agent in env.agents.values()}
    env.step(action_dict)
    env.render(fig=fig)
    x = []

