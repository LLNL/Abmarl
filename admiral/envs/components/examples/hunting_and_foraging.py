
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# Import all the features that we need from the simulation components
from admiral.envs.components.state import TeamState, GridPositionState, LifeState, GridResourceState
from admiral.envs.components.observer import PositionRestrictedPositionObserver, PositionRestrictedLifeObserver, PositionRestrictedTeamObserver, GridResourceObserver
from admiral.envs.components.actor import GridMovementActor, PositionTeamBasedAttackActor, GridResourcesActor
from admiral.envs.components.done import TeamDeadDone

# Environment needs a corresponding agent component
from admiral.envs.components.agent import TeamAgent, PositionAgent, LifeAgent, AgentObservingAgent, PositionObservingAgent, TeamObservingAgent, LifeObservingAgent, ResourceObservingAgent, GridMovementAgent, AttackingAgent, HarvestingAgent

# Import the interface
from admiral.envs import AgentBasedSimulation

# Import extra tools
from admiral.tools.matplotlib_utils import mscatter

# A generic agent, just for simplifying the coding. All agents
# have a position, team, and life/death state
# can observe positions, teams, and life state of other agents
# can move around the grid
class GenericAgent(PositionAgent, TeamAgent, GridMovementAgent, LifeAgent, PositionObservingAgent, TeamObservingAgent, LifeObservingAgent, AgentObservingAgent): pass

# Foraging agents can see the resources and harvest them
class ForagingAgent(GenericAgent, HarvestingAgent, ResourceObservingAgent):
    pass

# Hunting agents can attack other agents
class HuntingAgent(GenericAgent, AttackingAgent):
    pass

# Create the simulation environment from the components
class HuntingForagingEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        # Explicitly pull out the the dictionary of agents. This makes the step
        # function easier to work with.
        self.agents = kwargs['agents']

        # State components
        # These components track the state of the agents. This environment supports
        # agents with positions, life, and team.
        self.position_state = GridPositionState(**kwargs)
        self.life_state = LifeState(**kwargs)
        self.resource_state = GridResourceState(**kwargs)
        self.team_state = TeamState(**kwargs)

        # Observer components
        # These components handle the observations that the agents receive whenever
        # get_obs is called. In this environment supports agents that can observe
        # the position, health, and team of other agents and itself. It also supports
        # agents that can see resources. The observers are smart enough to know which
        # agents they can work with. For example, the resource observer will only
        # give resource observations to ForagingAgents.
        self.position_observer = PositionRestrictedPositionObserver(position=self.position_state, **kwargs)
        self.team_observer = PositionRestrictedTeamObserver(team=self.team_state, **kwargs)
        self.life_observer = PositionRestrictedLifeObserver(**kwargs)
        self.resource_observer = GridResourceObserver(resources=self.resource_state, **kwargs)

        # Actor components
        # These components handle the actions in the step function. This environment
        # supports agents that can move around, harvest, and attack agents from other teams.
        # The actors are smart enough to know which agents they can work with.
        # For example, the attack actor will only process attack actions from
        # HuntingAgents and the harvest actor will only process harvesting actions
        # from the ForagingAgent.
        self.move_actor = GridMovementActor(position=self.position_state, **kwargs)
        self.resource_actor = GridResourcesActor(resources=self.resource_state, **kwargs)
        self.attack_actor = PositionTeamBasedAttackActor(**kwargs)

        # Done components
        # This component tracks when the simulation is done. This environment is
        # done when either:
        # (1) All the hunter have killed all the foragers.
        # (2) All the foragers have harvested all the resources.
        # TODO: Need to implement this done condition!
        self.done = TeamDeadDone(**kwargs)

        # This is needed at the end of init in every environment. It ensures that
        # agents have been configured correctly.
        self.finalize()
    
    def reset(self, **kwargs):
        # The state handlers need to reset. Since the agents' teams do not change
        # throughout the episode, the team state does not need to reset.
        self.position_state.reset(**kwargs)
        self.resource_state.reset(**kwargs)
        self.life_state.reset(**kwargs)

    def step(self, action_dict, **kwargs):
        # Process harvesting
        for agent_id, action in action_dict.items():
            harvesting_agent = self.agents[agent_id]
            # The actor processes the agents' harvesting action. If there is a
            # resource on the same cell as this agent and the agent chooses to
            # harvest, then the resource will be harvested.
            harvested_amount = self.resource_actor.process_harvest(harvesting_agent, action.get('harvest', 0), **kwargs)
            if harvested_amount is not None:
                pass
        
        # Process attacking
        for agent_id, action in action_dict.items():
            attacking_agent = self.agents[agent_id]
            # The actor processes the agents' attack action. If an enemy agent is
            # within the attacker's attack radius, the attack will be successful.
            # If there are multiple enemy agents within the radius, the agent that
            # is attacked is randomly selected. The actor returns the agent
            # that has been attacked.
            attacked_agent = self.attack_actor.process_attack(attacking_agent, action.get('attack', False), **kwargs)
            # The attacked agent loses health depending on the strength of the attack.
            # If the agent loses all its health, it dies.
            if attacked_agent is not None:
                self.life_state.modify_health(attacked_agent, -attacking_agent.attack_strength)

        # Process movement
        for agent_id, action in action_dict.items():
            # The actor processes the agents' movement action. The agents can move
            # within their max move radius, and they can occupy the same cells.
            # If an agent attempts to move out of bounds, the move is invalid,
            # and it will not move at all.
            proposed_amount_move = action.get('move', np.zeros(2))
            amount_moved = self.move_actor.process_move(self.agents[agent_id], proposed_amount_move, **kwargs)
            if np.any(proposed_amount_move != amount_moved):
                # This was a rejected move, so we penalize a bit for it
                # self.rewards[agent_id] -= 0.1
                pass
    
    def render(self, fig=None, **kwargs):
        fig.clear()

        # Draw the resources
        ax = fig.gca()
        ax = sns.heatmap(np.flipud(self.resource_state.resources), ax=ax, cmap='Greens')

        # Draw the agents
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        shape_dict = {agent.id: 'o' if agent.team == 1 else 's' for agent in self.agents.values()}
        ax.set(xlim=(0, self.position_state.region), ylim=(0, self.position_state.region))
        ax.set_xticks(np.arange(0, self.position_state.region, 1))
        ax.set_yticks(np.arange(0, self.position_state.region, 1))
        ax.grid()

        agents_x = [agent.position[1] + 0.5 for agent in self.agents.values() if render_condition[agent.id]]
        agents_y = [self.position_state.region - 0.5 - agent.position[0] for agent in self.agents.values() if render_condition[agent.id]]
        shape = [shape_dict[agent_id] for agent_id in shape_dict if render_condition[agent_id]]
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return {
            **self.position_observer.get_obs(agent),
            **self.team_observer.get_obs(agent),
            **self.life_observer.get_obs(agent),
            **self.resource_observer.get_obs(agent),
        }
    
    def get_reward(self, agent_id, **kwargs):
        return 0

    def get_done(self, agent_id, **kwargs):
        return self.done.get_done(self.agents[agent_id])
    
    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)
    
    def get_info(self, *args, **kwargs):
        return {}

if __name__ == '__main__':
    foragers = {f'forager{i}': ForagingAgent(id=f'forager{i}', agent_view=5, team=0, move_range=1, min_harvest=1, max_harvest=1, resource_view=5) for i in range(7)}
    hunters =  {f'hunter{i}':  HuntingAgent( id=f'hunter{i}',  agent_view=2, team=1, move_range=1, attack_range=1, attack_strength=1) for i in range(2)}
    agents = {**foragers, **hunters}
    region = 10
    env = HuntingForagingEnv(
        region=region,
        agents=agents,
        number_of_teams=2,
        min_value=1.0,
        max_value=1.0
    )
    env.reset()
    import pprint; pprint.pprint({agent_id: env.get_obs(agent_id) for agent_id in env.agents})
    fig = plt.gcf()
    env.render(fig=fig)

    for _ in range(50):
        action_dict = {agent.id: agent.action_space.sample() for agent in env.agents.values() if agent.is_alive}
        env.step(action_dict)
        env.render(fig=fig)
        # print(env.get_all_done())
