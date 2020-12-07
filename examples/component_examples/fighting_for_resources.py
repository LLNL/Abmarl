
from matplotlib import pyplot as plt

from admiral.component_envs.observer import ObservingAgent
from admiral.component_envs.position import GridPositionComponent, GridPositionAgent
from admiral.component_envs.movement import GridMovementComponent, GridMovementAgent
from admiral.component_envs.resources import GridResourceComponent, GridResourceHarvestingAgent
from admiral.component_envs.attacking import GridAttackingComponent, GridAttackingAgent
from admiral.component_envs.death_life import DyingAgent, DyingComponent
from admiral.component_envs.rewarder import RewarderComponent
from admiral.component_envs.done_conditioner import DeadDoneComponent
from admiral.envs import AgentBasedSimulation

class FightForResourcesAgent(DyingAgent, GridPositionAgent, GridAttackingAgent, GridMovementAgent, GridResourceHarvestingAgent, ObservingAgent):
    pass

class FightForResourcesEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.position = GridPositionComponent(**kwargs)
        self.resource = GridResourceComponent(**kwargs)
        self.movement = GridMovementComponent(**kwargs)
        self.attacking = GridAttackingComponent(**kwargs)
        self.dying = DyingComponent(**kwargs)
        self.rewarder = RewarderComponent(**kwargs)
        self.done_conditioner = DeadDoneComponent(**kwargs)

        self.finalize()

        self.attacking_record = []
    
    def reset(self, **kwargs):
        self.position.reset(**kwargs)
        self.resource.reset(**kwargs)
        self.dying.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.is_alive:
                if action.get('attack', False):
                    attacked_agent = self.attacking.process_attack(agent)
                    if attacked_agent is not None:
                        self.agents[attacked_agent].health -= agent.attack_strength
                        self.attacking_record.append(agent.id + " attacked " + attacked_agent)
                if 'move' in action:
                    self.movement.process_move(agent, action['move'])
                if 'harvest' in action:
                    amount_harvested = self.resource.process_harvest(agent, action['harvest'])
                    agent.health += amount_harvested
            
        # Because agents can affect each others' health, we process the dying
        # outside the loop at the end of all the moves. Note: this does not
        # matter in a TurnBasedManager.
        for agent_id in action_dict:
            agent = self.agents[agent_id]
            if agent.is_alive:
                self.dying.apply_entropy(agent)
                self.dying.process_death(agent)
        
        self.resource.regrow()
    
    def render(self, fig=None, **kwargs):
        fig.clear()
        self.resource.render(fig=fig, **kwargs)
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        self.position.render(fig=fig, render_condition=render_condition, **kwargs)
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

agents = {f'agent{i}': FightForResourcesAgent(
    id=f'agent{i}', attack_range=1, attack_strength=0.4, move=1, max_harvest=1.0, view=3
) for i in range(6)}
env = FightForResourcesEnv(
    region=10,
    agents=agents
)
env.reset()
print({agent_id: env.get_obs(agent_id) for agent_id in env.agents})
fig = plt.gcf()
env.render(fig=fig)

for _ in range(50):
    action_dict = {}
    for agent_id, agent in env.agents.items():
        action_dict[agent_id] = {
            'move': agent.action_space['move'].sample(),
            'harvest': agent.action_space['harvest'].sample(),
            'attack': agent.action_space['attack'].sample(),
        }
    env.step(action_dict)
    env.render(fig=fig)
    print({agent_id: env.get_done(agent_id) for agent_id in env.agents})
    x = []

print(env.get_all_done())
