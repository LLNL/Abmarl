
class RewarderComponent:
    """
    Tracks the rewards for each agent throughout the simulation.
    """
    def __init__(self, agents=None, **kwargs):
        assert type(agents) is dict, "agents must be a dictionary"
        self.agents = agents
        self.rewards = {agent_id: 0.0 for agent_id in self.agents}

    def get_reward(self, agent_id, **kwargs):
        return self.rewards[agent_id]
