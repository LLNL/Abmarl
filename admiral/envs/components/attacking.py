
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

    position_state (PositionState):
        The attack is based on the distance between agents, so we need to query
        the position part of the state.
    
    life_state (LifeState):
        Successful attacks will change the health of the involved agents. Additionally,
        agents can only attack other agents that are still alive.

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

    # TODO: Can I do it by agent object instead of agent_id?
    def process_attack(self, attacking_agent, attack, **kwargs):
        """
        Determine which agent the attacking agent successfully attacks. Attacked
        agent's health will decrease, while attacking agent's health will increase
        by the attacking agent's attack_strength.

        attacking_agent (AttackingAgent):
            The agent that we are processing.

        attack (bool):
            True if the agent has chosen to attack, otherwise False.
        """
        if attack:
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

# TODO: there is so much duplication between this and the above class... Is there
# a way to use inheritance here to reduce duplication and still allow for the additional
# special case where the teams need to be the same?
class PositionTeamBasedAttackActor:
    """
    Provide the necessary action space for agents who can attack and process such
    attacks. The attack is successful if the attacked agent is alive, on a different
    team, and within range. The action space is appended with a MultiBinary(1),
    allowing the agent to attack or not attack.

    life_state (LifeState):
        Successful attacks will change the health of the involved agents. Additionally,
        agents can only attack other agents that are still alive.

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
        Determine which agent the attacking agent successfully attacks and return
        that agent's id. If the attack fails, return None.

        attacking_agent (AttackingAgetn):
            The agent that we are processing.

        attack (bool):
            True if the agent has chosen to attack, otherwise False.
        """
        if attack:
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
