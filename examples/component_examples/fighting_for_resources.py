
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from admiral.envs.components.observer import ObservingAgent
from admiral.envs.components.position import PositionState, PositionAgent, PositionObserver
from admiral.envs.components.movement import GridMovementActor, GridMovementAgent
from admiral.envs.components.resources import GridResourceState, GridResourceObserver, HarvestingAgent, GridResourcesActor
from admiral.envs.components.attacking import AttackingAgent, PositionBasedAttackActor
from admiral.envs.components.health import LifeAgent, LifeState, HealthObserver, LifeObserver
from admiral.envs.components.dead_done import DeadDone
from admiral.envs import AgentBasedSimulation
from admiral.tools.matplotlib_utils import mscatter

class FightForResourcesAgent(LifeAgent, PositionAgent, AttackingAgent, GridMovementAgent, HarvestingAgent, ObservingAgent):
    pass

class FightForResourcesEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State components
        self.position_state = PositionState(**kwargs)
        self.life_state = LifeState(**kwargs)
        self.resource_state = GridResourceState(**kwargs)

        # Observer components
        self.position_observer = PositionObserver(position=self.position_state, **kwargs)
        self.resource_observer = GridResourceObserver(resources=self.resource_state, **kwargs)
        self.health_observer = HealthObserver(**kwargs)
        self.life_observer = LifeObserver(**kwargs)

        # Actor components
        self.move_actor = GridMovementActor(position=self.position_state, **kwargs)
        self.resource_actor = GridResourcesActor(resources=self.resource_state, **kwargs)
        self.attack_actor = PositionBasedAttackActor(**kwargs)

        # Done components
        self.done = DeadDone(**kwargs)

        self.finalize()
    
    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.life_state.reset(**kwargs)
        self.resource_state.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        # Process harvesting
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            harvested_amount = self.resource_actor.process_harvest(agent, action['harvest'], **kwargs)
            if harvested_amount is not None:
                self.life_state.modify_health(agent, harvested_amount)
        
        # Process attacking
        for agent_id, action in action_dict.items():
            attacking_agent = self.agents[agent_id]
            attacked_agent = self.attack_actor.process_attack(attacking_agent, action['attack'], **kwargs)
            if attacked_agent is not None:
                self.life_state.modify_health(attacked_agent, -attacking_agent.attack_strength)

        # Process movement
        for agent_id, action in action_dict.items():
            self.move_actor.process_move(self.agents[agent_id], action['move'], **kwargs)

        # Apply entropy to all agents
        for agent_id in action_dict:
            self.life_state.apply_entropy(self.agents[agent_id])

        # Regrow the resources
        self.resource_state.regrow()
    
    def render(self, fig=None, **kwargs):
        fig.clear()

        # Draw the resources
        ax = fig.gca()
        ax = sns.heatmap(np.flipud(self.resource_state.resources), ax=ax, cmap='Greens')

        # Draw the agents
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        ax.set(xlim=(0, self.position_state.region), ylim=(0, self.position_state.region))
        ax.set_xticks(np.arange(0, self.position_state.region, 1))
        ax.set_yticks(np.arange(0, self.position_state.region, 1))
        ax.grid()

        agents_x = [agent.position[1] + 0.5 for agent in self.agents.values() if render_condition[agent.id]]
        agents_y = [self.position_state.region - 0.5 - agent.position[0] for agent in self.agents.values() if render_condition[agent.id]]
        mscatter(agents_x, agents_y, ax=ax, m='o', s=200, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return {
            'position': self.position_observer.get_obs(agent),
            'resources': self.resource_observer.get_obs(agent),
            'health': self.health_observer.get_obs(agent_id, **kwargs),
            'life': self.life_observer.get_obs(agent_id, **kwargs),
        }
    
    def get_reward(self, agent_id, **kwargs):
        pass

    def get_done(self, agent_id, **kwargs):
        return self.done.get_done(self.agents[agent_id])
    
    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)
    
    def get_info(self, **kwargs):
        return {}

agents = {f'agent{i}': FightForResourcesAgent(
    id=f'agent{i}', attack_range=1, attack_strength=0.4, move_range=1, max_harvest=1.0, view=3
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
    action_dict = {agent.id: agent.action_space.sample() for agent in env.agents.values() if agent.is_alive}
    env.step(action_dict)
    env.render(fig=fig)
    print({agent_id: env.get_done(agent_id) for agent_id in env.agents})
    x = []

print(env.get_all_done())
