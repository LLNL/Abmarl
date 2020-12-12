
from matplotlib import pyplot as plt

from admiral.envs.components.observer import ObservingAgent
from admiral.envs.components.position import PositionState, PositionObserver, PositionAgent
from admiral.envs.components.movement import GridMovementAgent, GridMovementActor
from admiral.envs.components.resources import GridResourceState, GridResourceObserver, HarvestingAgent, GridResourcesActor
from admiral.envs.components.health import LifeAgent, LifeState, HealthObserver, LifeObserver
from admiral.envs.components.dead_done import DeadDone
from admiral.envs import AgentBasedSimulation

class ResourceManagementAgent(LifeAgent, GridMovementAgent, PositionAgent, ObservingAgent,  HarvestingAgent):
    pass

class ResourceManagementEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State components
        self.position_state = PositionState(**kwargs)
        self.life_state = LifeState(**kwargs)
        self.resource_state = GridResourceState(**kwargs)

        # Observer components
        self.position_observer = PositionObserver(position=self.position_state, **kwargs)
        self.health_observer = HealthObserver(**kwargs)
        self.life_observer = LifeObserver(**kwargs)
        self.resource_observer = GridResourceObserver(resources=self.resource_state, **kwargs)

        # Actor components
        self.move_actor = GridMovementActor(position=self.position_state, **kwargs)
        self.resource_actor = GridResourcesActor(resources=self.resource_state, **kwargs)

        # Done components
        self.done = DeadDone(**kwargs)

        self.finalize()
    
    def reset(self, **kwargs):
        for agent in self.agents.values():
            self.position_state.reset(agent, **kwargs)
            self.resource_state.reset(**kwargs)
            self.life_state.reset(agent, **kwargs)
    
    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            harvested_amount = self.resource_actor.process_harvest(agent, action['harvest'], **kwargs)
            if harvested_amount is not None:
                self.life_state.modify_health(agent, harvested_amount)

        for agent_id, action in action_dict.items():
            self.move_actor.process_move(self.agents[agent_id], action['move'], **kwargs)

        # Apply entropy to all agents
        for agent_id, action in action_dict.items():
            self.life_state.modify_health(self.agents[agent_id], -0.1)

        # Regrow the resources
        self.resource_state.regrow()
    
    def render(self, fig=None, **kwargs):
        pass
        # fig.clear()
        # self.resource.render(fig=fig, **kwargs)
        # render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        # self.position.render(fig=fig, render_condition=render_condition, **kwargs)
        # plt.plot()
        # plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return {
            'position': self.position_observer.get_obs(agent),
            'resources': self.resource_observer.get_obs(agent),
        }
    
    def get_reward(self, agent_id, **kwargs):
        pass

    def get_done(self, agent_id, **kwargs):
        return self.done.get_done(self.agents[agent_id])
    
    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)
    
    def get_info(self, **kwargs):
        return {}

agents = {f'agent{i}': ResourceManagementAgent(id=f'agent{i}', view=2, move_range=1, max_harvest=1.0) for i in range(4)}
env = ResourceManagementEnv(
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
        if agent.is_alive:
            action_dict[agent_id] = agent.action_space.sample()
    env.step(action_dict)
    print({agent_id: env.get_done(agent_id) for agent_id in env.agents})
    env.render(fig=fig)

print(env.get_all_done())
