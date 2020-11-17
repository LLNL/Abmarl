
from abc import ABC, abstractmethod

from admiral.envs import Agent

class AttackingEnv(ABC):
    """
    AttackingEnv processes attack action, where agents can attack other agents.
    The attack is based on the agents' positions.
    """
    def __init__(self, agents=None, **kwargs):
        assert type(agents) is dict, "agents must be a dictionary"
        self.agents = agents

    @abstractmethod
    def process_attack(self, attacking_agent, **kwargs):
        """
        Process the attack from the attacking agent.
        """
        pass

class GridAttackingAgent(Agent):
    # TODO: Still exploring whether we should assert agent type...
    def __init__(self, attack_range=None, attack_strength=None, **kwargs):
        assert attack_range is not None, "attack_range must be a nonnegative integer"
        self.attack_range = attack_range
        assert attack_strength is not None, "attack_strength must be a nonnegative number"
        self.attack_strength = attack_strength
        super().__init__(**kwargs)

        from gym.spaces import MultiBinary
        self.action_space['attack'] = MultiBinary(1)
    
    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return super().configured and self.attack_range is not None and self.attack_strength is not None

class GridAttackingEnv(AttackingEnv):
    def process_attack(self, attacking_agent, **kwargs):
        for agent in self.agents.values():
            if abs(attacking_agent.position[0] - agent.position[0]) <= attacking_agent.attack_range \
                    and abs(attacking_agent.position[1] - agent.position[1]) <= attacking_agent.attack_range: # Agent within range
                return agent.id

class GridAttackingTeamEnv(AttackingEnv):
    # TODO: Rough design. Perhaps in the kwargs we should include a combination matrix that dictates
    # attacks that cannot happen?
    def process_attack(self, attacking_agent, **kwargs):
        for agent in self.agents.values():
            if agent.team == attacking_agent.team: continue # Cannot attack agents on same team
            if abs(attacking_agent.position[0] - agent.position[0]) <= attacking_agent.attack_range \
                    and abs(attacking_agent.position[1] - agent.position[1]) <= attacking_agent.attack_range: # Agent within range
                return agent.id
