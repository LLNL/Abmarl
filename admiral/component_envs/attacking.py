
from admiral.envs import Agent
from admiral.component_envs.world import GridWorldAgent, GridWorldTeamAgent

class GridAttackingAgent(GridWorldAgent):
    def __init__(self, attack_range=None, attack_strength=None, **kwargs):
        assert attack_range is not None, "attack_range must be a nonnegative integer"
        self.attack_range = attack_range
        assert attack_strength is not None, "attack_strength must be a nonnegative number"
        self.attack_strength = attack_strength
        super().__init__(**kwargs)
    
    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return super().configured and self.attack_range is not None and self.attack_strength is not None

class GridAttackingEnv:
    def __init__(self, agents=None, **kwargs):
        # assert type(agents) is dict, "agents must be a dict"
        # for agent in agents.values():
        #     assert isinstance(agent, GridAttackingAgent), "agents must be GridAttackingAgent"
        # self.agents = agents
        self.agents = {agent.id: agent for agent in agents.values() if isinstance(agent, GridAttackingAgent)}

        from gym.spaces import MultiBinary
        for agent in self.agents.values():
            agent.action_space['attack'] = MultiBinary(1)

    def process_attack(self, attacking_agent, **kwargs):
        for agent in self.agents.values():
            if agent.id == attacking_agent.id: continue # cannot attack yourself, lol
            if abs(attacking_agent.position[0] - agent.position[0]) <= attacking_agent.attack_range \
                    and abs(attacking_agent.position[1] - agent.position[1]) <= attacking_agent.attack_range: # Agent within range
                return agent.id

class GridAttackingTeamAgent(GridAttackingAgent, GridWorldTeamAgent):
    pass

class GridAttackingTeamEnv:
    # TODO: Rough design. Perhaps in the kwargs we should include a combination matrix that dictates
    # attacks that cannot happen?
    def __init__(self, agents=None, **kwargs):
        # assert type(agents) is dict, "agents must be a dict"
        # for agent in agents.values():
        #     assert isinstance(agent, GridAttackingTeamAgent), "agents must be GridAttackingAgent"
        # self.agents = agents
        self.agents = {agent.id: agent for agent in agents.values() if isinstance(agent, GridAttackingTeamAgent)}

        from gym.spaces import MultiBinary
        for agent in self.agents.values():
            agent.action_space['attack'] = MultiBinary(1)

    def process_attack(self, attacking_agent, **kwargs):
        for agent in self.agents.values():
            if agent.id == attacking_agent.id: continue # cannot attack yourself, lol
            if agent.team == attacking_agent.team: continue # Cannot attack agents on same team
            if abs(attacking_agent.position[0] - agent.position[0]) <= attacking_agent.attack_range \
                    and abs(attacking_agent.position[1] - agent.position[1]) <= attacking_agent.attack_range: # Agent within range
                return agent.id
