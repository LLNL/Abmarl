
from matplotlib import pyplot as plt
import numpy as np

# Import all the features that we need from the simulation components
from admiral.envs.components.state import GridPositionState, LifeState
from admiral.envs.components.observer import PositionObserver, LifeObserver, TeamObserver
from admiral.envs.components.wrappers.observer_wrapper import PositionRestrictedObservationWrapper
from admiral.envs.components.actor import GridMovementActor, AttackActor
from admiral.envs.components.done import AnyTeamDeadDone, TeamDeadDone

# Environment needs a corresponding agent component
from admiral.envs.components.agent import TeamAgent, PositionAgent, LifeAgent, AttackingAgent, GridMovementAgent, AgentObservingAgent, PositionObservingAgent, TeamObservingAgent, LifeObservingAgent

# Import the interface
from admiral.envs import AgentBasedSimulation

# Import extra tools
from admiral.tools.matplotlib_utils import mscatter

# All HuntingForagingAgents
# have a position, team, and life/death state
# can observe positions, teams, and life state of other agents
# can move around the grid and attack other agents
class HuntingForagingAgent(TeamAgent, PositionAgent, LifeAgent, AttackingAgent, GridMovementAgent, AgentObservingAgent, PositionObservingAgent, TeamObservingAgent, LifeObservingAgent): pass

# All FoodAgents
# have a tema, position, and life
# They are not really "agents" in the RL sense, they're just entities in the simulation for the foragers to gather.
class FoodAgent(TeamAgent, PositionAgent, LifeAgent): pass

# Create the simulation environment from the components
class HuntingForagingEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        # Explicitly pull out the the dictionary of agents. This makes the env
        # easier to work with.
        self.agents = kwargs['agents']

        # State components
        # These components track the state of the agents. This environment supports
        # agents with positions, life, and team.
        self.position_state = GridPositionState(**kwargs)
        self.life_state = LifeState(**kwargs)

        # Observer components
        # These components handle the observations that the agents receive whenever
        # get_obs is called. In this environment supports agents that can observe
        # the position, health, and team of other agents and itself.
        position_observer = PositionObserver(position=self.position_state, **kwargs)
        team_observer = TeamObserver(**kwargs)
        life_observer = LifeObserver(**kwargs)
        self.partial_observer = PositionRestrictedObservationWrapper([position_observer, team_observer, life_observer], **kwargs)

        # Actor components
        # These components handle the actions in the step function. This environment
        # supports agents that can move around and attack agents from other teams.
        self.move_actor = GridMovementActor(position=self.position_state, **kwargs)
        self.attack_actor = AttackActor(**kwargs)

        # Done components
        # This component tracks when the simulation is done. This environment is
        # done when either:
        # (1) All the hunter have killed all the foragers.
        # (2) All the foragers have killed all the resources.
        # self.done = AnyTeamDeadDone(**kwargs)
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
        # Process attacking
        for agent_id, action in action_dict.items():
            attacking_agent = self.agents[agent_id]
            # The actor processes the agents' attack action.
            attacked_agent = self.attack_actor.process_attack(attacking_agent, action.get('attack', False), **kwargs)
            # The attacked agent loses health depending on the strength of the attack.
            # If the agent loses all its health, it dies.
            if attacked_agent is not None:
                self.life_state.modify_health(attacked_agent, -attacking_agent.attack_strength)
                # Reward the attacking agent for its successful attack
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
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=100, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return self.partial_observer.get_obs(agent, **kwargs)
    
    def get_reward(self, agent_id, **kwargs):
        """
        Return the agents reward and reset it to zero.
        """
        reward_out = self.rewards[agent_id]
        self.rewards[agent_id] = 0
        return reward_out

    def get_done(self, agent_id, **kwargs):
        return self.done.get_done(self.agents[agent_id], **kwargs)
    
    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)
    
    def get_info(self, *args, **kwargs):
        return {}

if __name__ == '__main__':
    food = {f'food{i}': FoodAgent(id=f'food{i}', team=1) for i in range(12)}
    foragers = {f'forager{i}': HuntingForagingAgent(id=f'forager{i}', agent_view=5, team=2, move_range=1, attack_range=1, attack_strength=1) for i in range(7)}
    hunters =  {f'hunter{i}':  HuntingForagingAgent(id=f'hunter{i}',  agent_view=2, team=3, move_range=1, attack_range=1, attack_strength=1) for i in range(2)}
    agents = {**food, **foragers, **hunters}

    region = 20
    team_attack_matrix = np.zeros((4, 4))
    team_attack_matrix[2, 1] = 1
    team_attack_matrix[3, 2] = 1
    env = HuntingForagingEnv(
        region=region,
        agents=agents,
        team_attack_matrix=team_attack_matrix,
        number_of_teams=3,
    )
    env.reset()

    shape_dict = {
        1: 's',
        2: 'o',
        3: 'd'
    }

    import pprint; pprint.pprint({agent_id: env.get_obs(agent_id) for agent_id in env.agents})
    fig = plt.gcf()
    env.render(fig=fig, shape_dict=shape_dict)

    for _ in range(50):
        action_dict = {agent.id: agent.action_space.sample() for agent in env.agents.values() if agent.is_alive and isinstance(agent, HuntingForagingAgent)}
        env.step(action_dict)
        env.render(fig=fig, shape_dict=shape_dict)
        print(env.get_all_done())
