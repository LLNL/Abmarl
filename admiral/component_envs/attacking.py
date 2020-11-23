
from admiral.envs import Agent
from admiral.component_envs.world import GridWorldAgent, GridWorldTeamAgent

def GridWorldAttackingAgent(attack_range=None, attack_strength=None, **kwargs):
    return {
        **GridWorldAgent(**kwargs),
        'attack_range': attack_range,
        'attack_strength': attack_strength,
    }

class GridAttackingEnv:
    def __init__(self, agents=None, **kwargs):
        assert type(agents) is dict, "agents must be a dict"
        self.agents = agents

        from gym.spaces import MultiBinary
        for agent in self.agents.values():
            if 'attack_range' in agent and 'attack_strength' in agent:
                agent['action_space']['attack'] = MultiBinary(1)

    def process_attack(self, attacking_agent, **kwargs):
        if 'attack_range' in attacking_agent and 'attack_strength' in attacking_agent:
            for agent in self.agents.values():
                if agent.id == attacking_agent.id: continue # cannot attack yourself, lol
                if abs(attacking_agent.position[0] - agent.position[0]) <= attacking_agent.attack_range \
                        and abs(attacking_agent.position[1] - agent.position[1]) <= attacking_agent.attack_range: # Agent within range
                    return agent.id

def GridWorldAttackingTeamAgent(**kwargs):
    return {
        **GridWorldAttackingAgent(**kwargs),
        **GridWorldTeamAgent(**kwargs),
    }

class GridAttackingTeamEnv:
    # TODO: Rough design. Perhaps in the kwargs we should include a combination matrix that dictates
    # attacks that cannot happen?
    def __init__(self, agents=None, **kwargs):
        assert type(agents) is dict, "agents must be a dict"
        self.agents = agents

        from gym.spaces import MultiBinary
        for agent in self.agents.values():
            if 'attack_range' in agent and 'attack_strength' in agent and 'team' in agent:
                agent.action_space['attack'] = MultiBinary(1)

    def process_attack(self, attacking_agent, **kwargs):
        if 'attack_range' in attacking_agent and 'attack_strength' in attacking_agent and 'team' in attacking_agent:
            for agent in self.agents.values():
                if agent.id == attacking_agent.id: continue # cannot attack yourself, lol
                if agent.team == attacking_agent.team: continue # Cannot attack agents on same team
                if abs(attacking_agent.position[0] - agent.position[0]) <= attacking_agent.attack_range \
                        and abs(attacking_agent.position[1] - agent.position[1]) <= attacking_agent.attack_range: # Agent within range
                    return agent.id
