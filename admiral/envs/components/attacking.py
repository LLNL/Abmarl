
from admiral.envs import Agent
from admiral.envs.components.position import GridPositionAgent
from admiral.envs.components.team import TeamAgent
from admiral.envs.components.death_life import LifeAgent

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

class PositionBasedHealthExchangeAttackActor:
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
    def __init__(self, position_state=None, life_state=None, agents=None, **kwargs):
        self.position_state = position_state
        self.life_state = life_state
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
    def process_attack(self, attacking_agent_id, attack, **kwargs):
        """
        Determine which agent the attacking agent successfully attacks. Attacked
        agent's health will decrease, while attacking agent's health will increase
        by the attacking agent's attack_strength.

        attacking_agent_id (str):
            The agent that we are processing.

        attack (bool):
            True if the agent has chosen to attack, otherwise False.
        """
        if attack:
            attacking_agent = self.agents[attacking_agent_id]
            for attacked_agent_id, attacked_agent in self.agents.items():
                if attacked_agent_id == attacking_agent_id:
                    # Cannot attack yourself
                    continue
                # TODO: This is super verbose.... Maybe I should avoid using the
                # get state and only take the component that I need to modify?
                # Queries can just be done on the agent objects themselves.
                elif not self.life_state.get_state(attacked_agent_id):
                    # Cannot attack dead agents
                    continue
                elif abs(self.position_state.get_state(attacked_agent_id)[0] - self.position_state.get_state(attacking_agent_id)[0]) > attacking_agent.attack_range or \
                     abs(self.position_state.get_state(attacked_agent_id)[1] - self.position_state.get_state(attacking_agent_id)[1]) > attacking_agent.attack_range:
                    # Agent too far away
                    continue
                else:
                    # Agent was successfully attacked
                    # TODO: HealthExchangeActor can process the actual change of
                    # health state in the agents. So then we can use the same AttackActor
                    # but with a different HealthExchangeActor for different effects,
                    # such as cases where we only want the health to decrease for
                    # the attacked agent.
                    self.life_state.modify_health(attacked_agent_id, -attacking_agent.attack_strength, **kwargs)
                    self.life_state.modify_health(attacking_agent_id, attacking_agent.attack_strength, **kwargs)

class PositionTeamBasedHealthExchangeAttackActor:
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
    def __init__(self, life_state=None, agents=None, **kwargs):
        self.life_state=life_state
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

    def process_attack(self, attacking_agent_id, attack, **kwargs):
        """
        Determine which agent the attacking agent successfully attacks and return
        that agent's id. If the attack fails, return None.

        attacking_agent_id (str):
            The agent that we are processing.

        attack (bool):
            True if the agent has chosen to attack, otherwise False.
        """
        if attack:
            attacking_agent = self.agents[attacking_agent_id]
            for attacked_agent in self.agents.values():
                if attacked_agent.id == attacking_agent_id:
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
                    self.life_state.modify_health(attacked_agent.id, -attacking_agent.attack_strength, **kwargs)
                    self.life_state.modify_health(attacking_agent_id, attacking_agent.attack_strength, **kwargs)
