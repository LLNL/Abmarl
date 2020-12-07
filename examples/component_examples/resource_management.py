
from matplotlib import pyplot as plt

from admiral.component_envs.observer import ObservingAgent
from admiral.component_envs.position import GridPositionComponent, GridPositionAgent
from admiral.component_envs.movement import GridMovementComponent, GridMovementAgent
from admiral.component_envs.resources import GridResourceComponent, GridResourceHarvestingAgent
from admiral.component_envs.death_life import LifeAgent, DyingComponent
from admiral.component_envs.rewarder import RewarderComponent
from admiral.component_envs.done_component import DeadDoneComponent
from admiral.envs import AgentBasedSimulation

class ResourceManagementAgent(LifeAgent, GridMovementAgent, GridPositionAgent, ObservingAgent,  GridResourceHarvestingAgent):
    pass

class ResourceManagementEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.position = GridPositionComponent(**kwargs)
        self.resource = GridResourceComponent(**kwargs)
        self.movement = GridMovementComponent(**kwargs)
        self.dying = DyingComponent(**kwargs)
        self.rewarder = RewarderComponent(**kwargs)
        self.done_conditioner = DeadDoneComponent(**kwargs)

        self.finalize()
    
    def reset(self, **kwargs):
        self.position.reset(**kwargs)
        self.resource.reset(**kwargs)
        self.dying.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.is_alive:
                if 'move' in action:
                    self.movement.process_move(agent, action['move'])
                if 'harvest' in action:
                    amount_harvested = self.resource.process_harvest(agent, action['harvest'])
                    agent.add_health(amount_harvested)
                self.dying.apply_entropy(agent)
        self.resource.regrow()
    
    def render(self, fig=None, **kwargs):
        fig.clear()
        self.resource.render(fig=fig, **kwargs)
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        self.position.render(fig=fig, render_condition=render_condition, **kwargs)
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

agents = {f'agent{i}': ResourceManagementAgent(id=f'agent{i}', view=2, move=1, max_harvest=1.0) for i in range(4)}
env = ResourceManagementEnv(
    region=10,
    agents=agents
)
env.reset()
print({agent_id: env.get_obs(agent_id) for agent_id in env.agents})
fig = plt.gcf()
env.render(fig=fig)

for _ in range(50):
    action_dict = {agent_id: {} for agent_id in env.agents}
    for agent_id, agent in env.agents.items():
        action_dict[agent_id] = {
            'move': agent.action_space['move'].sample(),
            'harvest': agent.action_space['harvest'].sample()
        }
    env.step(action_dict)
    print({agent_id: env.get_done(agent_id) for agent_id in env.agents})
    env.render(fig=fig)

print(env.get_all_done())
