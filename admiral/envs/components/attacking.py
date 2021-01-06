
from admiral.envs import Agent
from admiral.envs.components.agent import AttackingAgent
from admiral.envs.components.agent import PositionAgent
from admiral.envs.components.agent import TeamAgent
from admiral.envs.components.agent import LifeAgent


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
