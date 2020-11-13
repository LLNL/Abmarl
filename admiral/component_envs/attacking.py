
from abc import ABC, abstractmethod

from .world import WorldAgent

class AttackingTeamAgent(WorldAgent):
    def __init__(self, team=None, attack_range=None, **kwargs):
        super().__init__(**kwargs)
        self.team = team
        self.attack_range = attack_range
    
    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return super().configured and self.team is not None and self.attack_range is not None

class AttackingEnv(ABC):
    """
    AttackingEnv processes attack action, where agents can attack other agents.
    The attack is based on the agents' positions.
    """
    def __init__(self, agents=None):
        assert type(agents) is dict, "agents must be a dictionary"
        for agent in self.agents.values():
            assert isinstance(agent, AttackingTeamAgent)

    @abstractmethod
    def process_attack(self, attacking_agent, **kwargs):
        pass

class GridAttackingEnv(AttackingEnv):
    def process_attack(self, attacking_agent, **kwargs):
        for agent in self.agents.values():
            if agent.team == attacking_agent.team: continue # Cannot attack agents on same team
            if abs(attacking_agent.position[0] - agent.position[0]) <= attacking_agent.attack_range \
                    and abs(attacking_agent.position[1] - agent.position[1]) <= attacking_agent.attack_range: # Agent within range
                return agent.id
