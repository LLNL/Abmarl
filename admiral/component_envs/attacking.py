
from admiral.envs import Agent
from admiral.component_envs.position import GridPositionAgent
from admiral.component_envs.team import TeamAgent
from admiral.component_envs.death_life import LifeAgent

class GridAttackingAgent(Agent):
    def __init__(self, attack_range=None, attack_strength=None, **kwargs):
        super().__init__(**kwargs)
        assert attack_range is not None, "attack_range must be a nonnegative integer"
        self.attack_range = attack_range
        assert attack_strength is not None, "attack_strength must be a nonnegative number"
        self.attack_strength = attack_strength
    
    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return super().configured and self.attack_range is not None and self.attack_strength is not None

class GridAttackingComponent:
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
