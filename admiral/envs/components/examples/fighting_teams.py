
from matplotlib import pyplot as plt
import numpy as np

# Import all the features that we need from the simulation components
from admiral.envs.components.state import GridPositionState, LifeState
from admiral.envs.components.observer import TeamObserver, PositionObserver, LifeObserver
from admiral.envs.components.wrappers.observer_wrapper import PositionRestrictedObservationWrapper
from admiral.envs.components.actor import GridMovementActor, AttackActor
from admiral.envs.components.done import TeamDeadDone
# Each env component needs a corresponding agent component
from admiral.envs.components.agent import TeamAgent, PositionAgent, LifeAgent, TeamObservingAgent, PositionObservingAgent, LifeObservingAgent, AgentObservingAgent, GridMovementAgent, AttackingAgent

# Import the interface
from admiral.envs import AgentBasedSimulation

# Import extra tools
from admiral.tools.matplotlib_utils import mscatter

# Define a battle agent's attributes and capabilities
# These agents have a position, team, life/death state
# These agents can observe the above attributes of other agents and itself.
# These agents can move around in the grid and attack other agents
class FightingTeamAgent(TeamAgent, PositionAgent, LifeAgent, TeamObservingAgent, PositionObservingAgent, LifeObservingAgent, AgentObservingAgent, GridMovementAgent, AttackingAgent):
    pass

# Create the simulation environment from the components
class FightingTeamsEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        # Explicitly pull out the the dictionary of agents. This makes the step
        # function easier to work with.
        self.agents = kwargs['agents']

        # State Components
        # These components track the state of the agents. This environment supports
        # agents with positions, life, and team.
        self.position_state = GridPositionState(**kwargs)
        self.life_state = LifeState(**kwargs)

        # Observer Components
        # These components handle the observations that the agents receive whenever
        # get_obs is called. In this environment supports agents that can observe
        # the position, health, and team of other agents and itself. We then filter
        # those observations based on our partial observation filter settings.
        position_observer = PositionObserver(position=self.position_state, **kwargs)
        life_observer = LifeObserver(**kwargs)
        team_observer = TeamObserver(**kwargs)
        self.observer = PositionRestrictedObservationWrapper([position_observer, life_observer, team_observer], **kwargs)

        # Actor Components
        # These components handle the actions in the step function. This environment
        # supports agents that can move around and attack agents from other teams.
        self.move_actor = GridMovementActor(position=self.position_state, **kwargs)
        self.attack_actor = AttackActor(**kwargs)

        # Done components
        # This component tracks when the simulation is done. This env is done when
        # all the agents remaining are all on the same team.
        self.done = TeamDeadDone(**kwargs)

        # This is needed at the end of init in every environment. It ensures that
        # agents have been configured correctly.
        self.finalize()
    
    def reset(self, **kwargs):
        # The state handlers need to reset. Since the agents' teams do not change
        # throughout the episode, the team state does not need to reset.
        self.position_state.reset(**kwargs)
        self.life_state.reset(**kwargs)

        # We haven't designed the rewards handling yet, so we'll do it manually for now.
        # An important principle to follow in MARL: track the rewards of all the agents
        # and report them when get_reward is called. Once the reward is reported,
        # reset it to zero.
        self.rewards = {agent: 0 for agent in self.agents}
    
    def step(self, action_dict, **kwargs):

        # # Hack for team 2
        # for agent_id in action_dict:
        #     agent = self.agents[agent_id]
        #     if agent.team == 2:
        #         action_dict[agent_id] = agent.action_space.sample()

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
                # Reward the agents according to the outcome of the actions
                self.rewards[attacked_agent.id] -= 1
                self.rewards[attacking_agent.id] += 1
    
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
                self.rewards[agent_id] -= 0.1
        
        # Small penalty for every agent that acted in this time step to incentive rapid actions
        for agent_id in action_dict:
            self.rewards[agent_id] -= 0.01

        # Hack for team 2 no reward
        for agent in self.agents.values():
            if agent.team == 2:
                self.rewards[agent.id] = 0
    
    def render(self, fig=None, shape_dict=None, **kwargs):
        fig.clear()
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}

        ax = fig.gca()
        ax.set(xlim=(0, self.position_state.region), ylim=(0, self.position_state.region))
        ax.set_xticks(np.arange(0, self.position_state.region, 1))
        ax.set_yticks(np.arange(0, self.position_state.region, 1))
        ax.grid()

        agents_x = [agent.position[1] + 0.5 for agent in self.agents.values() if render_condition[agent.id]]
        agents_y = [self.position_state.region - 0.5 - agent.position[0] for agent in self.agents.values() if render_condition[agent.id]]

        if shape_dict:
            shape = [shape_dict[agent.team] for agent in self.agents.values() if render_condition[agent.id]]
        else:
            shape = 'o'
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return self.observer.get_obs(agent, **kwargs)
    
    def get_reward(self, agent_id, **kwargs):
        """
        Return the agents reward and reset it to zero.
        """
        reward_out = self.rewards[agent_id]
        self.rewards[agent_id] = 0
        return reward_out

    def get_done(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return self.done.get_done(agent)
    
    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)
    
    def get_info(self, *args, **kwargs):
        return {}

# Running this script via python3 battle_env will generate a simulation with three
# teams battling, each agent taking random actions.
if __name__ == '__main__':
    agents = {f'agent{i}': FightingTeamAgent(
        id=f'agent{i}', attack_range=1, attack_strength=0.4, team=i%2+1, move_range=1, agent_view=2
    ) for i in range(24)}
    env = FightingTeamsEnv(
        region=12,
        agents=agents,
        number_of_teams=2
    )
    env.reset()

    import pprint; pprint.pprint({agent_id: env.get_obs(agent_id) for agent_id in env.agents})
    fig = plt.gcf()

    shape_dict = {0: 's', 1: 'o'}
    env.render(fig=fig, shape_dict=shape_dict)

    for _ in range(100):
        action_dict = {agent.id: agent.action_space.sample() for agent in env.agents.values() if agent.is_alive}
        env.step(action_dict)
        env.render(fig=fig, shape_dict=shape_dict)
