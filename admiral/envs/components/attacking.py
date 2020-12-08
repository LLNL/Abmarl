
from admiral.envs import Agent
from admiral.envs.components.position import GridPositionAgent
from admiral.envs.components.team import TeamAgent
from admiral.envs.components.death_life import LifeAgent

class GridAttackingAgent(Agent):
    """
    Agents that can attack other agents in the grid. Attack is based on the relative
    positions of the agents and can affect the agents' health, if applicable.

    attack_range (int):
        The effective range of the attack.
    
    attack_strength (float):
        How effective the agent's attack is. This is applicable in situations where
        the agents' health is affected by attacks.
    """
    def __init__(self, attack_range=None, attack_strength=None, **kwargs):
        super().__init__(**kwargs)
        assert attack_range is not None, "attack_range must be a nonnegative integer"
        self.attack_range = attack_range
        assert attack_strength is not None, "attack_strength must be a nonnegative number"
        self.attack_strength = attack_strength
    
    @property
    def configured(self):
        """
        The agent is successfully configured if the attack range and strength is
        specified.
        """
        return super().configured and self.attack_range is not None and self.attack_strength is not None

class GridAttackingComponent:
    """
    Provide the necessary action space for agents who can attack and process such
    attacks. The attack is successful if the attacked agent is alive and within
    range. The action space is appended with a MultiBinary(1), allowing the agent
    to attack or not attack.

    agents (dict):
        The dictionary of agents. Because attacks are grid-based, all agents must
        be GridPositionAgents; becuase the attacked agent must be alive, all agents
        must be LifeAgents.
    """
    def __init__(self, agents=None, **kwargs):
        assert type(agents) is dict, "agents must be a dict"
        for agent in agents.values():
            assert isinstance(agent, GridPositionAgent)
            assert isinstance(agent, LifeAgent)
        self.agents = agents

        from gym.spaces import MultiBinary
        for agent in self.agents.values():
            if isinstance(agent, GridAttackingAgent):
                agent.action_space['attack'] = MultiBinary(1)

    def process_attack(self, attacking_agent, **kwargs):
        """
        Determine which agent the attacking agent successfully attacks and return
        that agent's id. If the attack fails, return None.

        attacking_agent (agent):
            The agent that has chosen to attack.
        """
        for agent in self.agents.values():
            if agent.id == attacking_agent.id:
                # Cannot attack yourself
                continue
            elif not agent.is_alive:
                # Cannot attack dead agents
                continue
            elif abs(attacking_agent.position[0] - agent.position[0]) > attacking_agent.attack_range or \
               abs(attacking_agent.position[1] - agent.position[1]) > attacking_agent.attack_range:
                # Agent too far away
                continue
            else:
                # Agent was successfully attacked
                return agent.id

class GridAttackingTeamComponent:
    """
    Provide the necessary action space for agents who can attack and process such
    attacks. The attack is successful if the attacked agent is alive, on a different
    team, and within range. The action space is appended with a MultiBinary(1),
    allowing the agent to attack or not attack.

    agents (dict):
        The dictionary of agents. Because attacks are grid-based, all agents must
        be GridPositionAgents; becuase the attacked agent must be alive, all agents
        must be LifeAgents; and because the attacked agent must be on a different
        team, each agent must be a TeamAgent.
    """
    def __init__(self, agents=None, **kwargs):
        assert type(agents) is dict, "agents must be a dict"
        for agent in agents.values():
            assert isinstance(agent, GridPositionAgent)
            assert isinstance(agent, TeamAgent)
            assert isinstance(agent, LifeAgent)
        self.agents = agents

        from gym.spaces import MultiBinary
        for agent in self.agents.values():
            if isinstance(agent, GridAttackingAgent):
                agent.action_space['attack'] = MultiBinary(1)

    def process_attack(self, attacking_agent, **kwargs):
        """
        Determine which agent the attacking agent successfully attacks and return
        that agent's id. If the attack fails, return None.

        attacking_agent (agent):
            The agent that has chosen to attack.
        """
        for agent in self.agents.values():
            if agent.id == attacking_agent.id:
                # Cannot attack yourself
                continue
            elif not agent.is_alive:
                # Cannot attack dead agents
                continue
            elif attacking_agent.team == agent.team:
                # Cannot attack agents on the same team
                continue
            elif abs(attacking_agent.position[0] - agent.position[0]) > attacking_agent.attack_range or \
               abs(attacking_agent.position[1] - agent.position[1]) > attacking_agent.attack_range:
                # Agent too far away
                continue
            else:
                # Agent was successfully attacked
                return agent.id
