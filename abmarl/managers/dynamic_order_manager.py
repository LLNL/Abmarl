
from abmarl.sim import AgentBasedSimulation
from abmarl.managers import SimulationManager

class DynamicOrderSimulation(AgentBasedSimulation):
    @property
    def next_agent(self):
        return self._next_agent

    @next_agent.setter
    def next_agent(self, value):
        assert type(value) is str, "Next agent type must be a string."
        assert value in self.agents, "Next agent must be an agent in the simulation."
        self._next_agent = value
    
    @property
    def just_done(self):
        return self._just_done
    
    @just_done.setter
    def just_done(self, value):
        assert type(value) is set, "Just done agents must be a set of agents."
        self._just_done = value

class DynamicOrderManager(SimulationManager):
    def __init__(self, sim):
        assert isinstance(sim, DynamicOrderSimulation), "Simulation must be a DynamicOrderSimulation."
        super().__init__(sim)
    
    def reset(self, **kwargs):
        self.done_agents = set()
        self.sim.reset(**kwargs)
        next_agent = self.sim.next_agent
        return {next_agent: self.sim.get_obs(next_agent)}
    
    def step(self, action_dict, **kwargs):
        agent_id = next(iter(action_dict))
        assert agent_id not in self.done_agents, \
            "Received an action for an agent that is already done."
        self.sim.just_done.clear()
        self.sim.step(action_dict, **kwargs)
        
        obs, rewards, dones, infos = {}, {}, {'__all__': self.sim.get_all_done()}, {}
        if dones['__all__']: # The simulation is done. Get output for all non-done agents
            # NOTE: you may not want to do this....
            for agent in self.agents:
                if agent in self.done_agents:
                    continue
                else:
                    obs[agent] = self.sim.get_obs(agent)
                    rewards[agent] = self.sim.get_reward(agent)
                    dones[agent] = self.sim.get_done(agent)
                    infos[agent] = self.sim.get_info(agent)
        else: # Simulation is not done. Get the output for the next agent and all agents who recently finished
            # All agents who finished in this step
            for agent in self.sim.just_done:
                assert agent not in self.done_agents, f"{agent.id} marked as just finished, but it had previously been marked as finished."
                obs[agent] = self.sim.get_obs(agent)
                rewards[agent] = self.sim.get_reward(agent)
                dones[agent] = self.sim.get_done(agent)
                infos[agent] = self.sim.get_info(agent)
                self.done_agents.add(agent)

            # The agent expected to send an action in the next step
            next_agent = self.sim.next_agent
            assert next_agent not in self.done_agents, "Next agent must be not done."
            obs[next_agent] = self.sim.get_obs(next_agent)
            rewards[next_agent] = self.sim.get_reward(next_agent)
            dones[next_agent] = self.sim.get_done(next_agent)
            infos[next_agent] = self.sim.get_info(next_agent)

        return obs, rewards, dones, infos


if __name__ == '__main__':
    import random
    from abmarl.sim import PrincipleAgent
    class ExampleSim(DynamicOrderSimulation):
        def __init__(self, agents=None, **kwargs):
            self.agents = agents
        
        def reset(self, **kwargs):
            self.just_done = set()
            self.next_agent = random.choice(list(self.agents.keys()))
        
        def step(self, action_dict, **kwargs):
            # An example where the next agent is chosen randomly. In your implementation,
            # you just need to set self.next_agent to the agent who should
            # be sending an action in the next step. This agent must be not done.
            self.next_agent = random.choice(list(self.agents.keys()))
        
        def render(self, **kwargs):
            pass
        
        def get_obs(self, agent_id, **kwargs):
            return f"Hello from agent {agent_id}"
        
        def get_reward(self, agent_id, **kwargs):
            return {}
        
        def get_done(self, agent_id, **kwargs):
            return {}
        
        def get_all_done(self, **kwargs):
            return {}

        def get_info(self, agent_id, **kwargs):
            return {}
    
    agents = {f'agent{i}': PrincipleAgent(id=f'agent{i}') for i in range(12)}
    sim = DynamicOrderManager(ExampleSim(agents=agents))
    obs = sim.reset()
    print(obs)
    for _ in range(24):
        obs, *_ = sim.step({'dummy_string'})
        print(obs)
