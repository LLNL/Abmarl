
from admiral.envs import Agent
from admiral.envs.components.position import PositionAgent
from admiral.envs.components.team import TeamAgent
from admiral.envs.components.health import LifeAgent

class AttackingAgent(Agent):
    """
    Agents that can attack other agents.

    attack_range (int):
        The effective range of the attack. Can be used to determine if an attack
        is successful based on distance between agents.
    
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

class PositionBasedAttackActor:
    """
    Provide the necessary action space for agents who can attack and processes such
    attacks. The attack is successful if the attacked agent is alive and within
    range. The action space is appended with a MultiBinary(1), allowing the agent
    to attack or not attack.

    agents (dict):
        The dictionary of agents. Because attacks are distance-based, all agents must
        be PositionAgents; becuase the attacked agent must be alive, all agents
        must be LifeAgents.
    """
    def __init__(self, agents=None, **kwargs):
        assert type(agents) is dict, "agents must be a dict"
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
            assert isinstance(agent, LifeAgent)
        self.agents = agents

        from gym.spaces import MultiBinary
        for agent in self.agents.values():
            if isinstance(agent, AttackingAgent):
                agent.action_space['attack'] = MultiBinary(1)

    def process_attack(self, attacking_agent, attack, **kwargs):
        """
        Determine which agent the attacking agent successfully attacks.

        attacking_agent (AttackingAgent):
            The agent that we are processing.

        attack (bool):
            True if the agent has chosen to attack, otherwise False.

        return (Agent):
            Return the attacked agent object. This can be None if no agent was
            attacked.
        """
        if isinstance(attacking_agent, AttackingAgent) and attack:
            for attacked_agent in self.agents.values():
                if attacked_agent.id == attacking_agent.id:
                    # Cannot attack yourself
                    continue
                elif not attacked_agent.is_alive:
                    # Cannot attack dead agents
                    continue
                elif abs(attacking_agent.position[0] - attacked_agent.position[0]) > attacking_agent.attack_range or \
                    abs(attacking_agent.position[1] - attacked_agent.position[1]) > attacking_agent.attack_range:
                    # Agent too far away
                    continue
                else:
                    return attacked_agent

class PositionTeamBasedAttackActor:
    """
    Provide the necessary action space for agents who can attack and process such
    attacks. The attack is successful if the attacked agent is alive, on a different
    team, and within range. The action space is appended with a MultiBinary(1),
    allowing the agent to attack or not attack.

    agents (dict):grid
        The dictionary of agents. Because attacks are distance-based, all agents must
        be PositionAgents; because the attacked agent must be alive, all agents
        must be LifeAgents; and because the attacked agent must be on a different
        team, all agents must be TeamAgents.
    """
    def __init__(self, agents=None, **kwargs):
        assert type(agents) is dict, "agents must be a dict"
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
            assert isinstance(agent, TeamAgent)
            assert isinstance(agent, LifeAgent)
        self.agents = agents

        from gym.spaces import MultiBinary
        for agent in self.agents.values():
            if isinstance(agent, AttackingAgent):
                agent.action_space['attack'] = MultiBinary(1)

    def process_attack(self, attacking_agent, attack, **kwargs):
        """
        Determine which agent the attacking agent successfully attacks.

        attacking_agent (AttackingAgent):
            The agent that we are processing.

        attack (bool):
            True if the agent has chosen to attack, otherwise False.

        return (Agent):
            Return the attacked agent object.
        """
        if isinstance(attacking_agent, AttackingAgent) and attack:
            for attacked_agent in self.agents.values():
                if attacked_agent.id == attacking_agent.id:
                    # Cannot attack yourself
                    continue
                elif not attacked_agent.is_alive:
                    # Cannot attack dead agents
                    continue
                elif attacking_agent.team == attacked_agent.team:
                    # Cannot attack agents on the same team
                    continue
                elif abs(attacking_agent.position[0] - attacked_agent.position[0]) > attacking_agent.attack_range or \
                     abs(attacking_agent.position[1] - attacked_agent.position[1]) > attacking_agent.attack_range:
                    # Agent too far away
                    continue
                else:
                    return attacked_agent
