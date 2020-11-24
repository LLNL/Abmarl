
class DoneConditioner:
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
        self.dones = {agent_id: False for agent_id in self.agents}

    def get_done(self, agent_id, **kwargs):
        return self.dones[agent_id]
    
    def get_all_done(self, **kwargs):
        return all([self.get_done(agent_id) for agent_id in self.agents])
    