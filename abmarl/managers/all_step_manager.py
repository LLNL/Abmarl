from .simulation_manager import SimulationManager


class AllStepManager(SimulationManager):
    """
    The AllStepManager gets the observations of all agents at reset. At step, it gets
    the observations of all the agents that are not done. Once all the agents
    are done, the manager returns all done.
    """
    def reset(self, **kwargs):
        """
        Reset the simulation and return the observation of all the agents.
        """
        self.done_agents = set()
        self.sim.reset(**kwargs)
        return {agent.id: self.sim.get_obs(agent.id) for agent in self.agents.values()}

    def step(self, action_dict, **kwargs):
        """
        Assert that the incoming action does not come from an agent who is recorded
        as done. Step the simulation forward and return the observation, reward,
        done, and info of all the non-done agents, including the agents that were
        done in this step. If all agents are done in this turn, then the manager
        returns all done.
        """
        for agent_id in action_dict:
            assert agent_id not in self.done_agents, \
                "Received an action for an agent that is already done."
        self.sim.step(action_dict, **kwargs)

        obs = {
            agent.id: self.sim.get_obs(agent.id) for agent in self.agents.values()
            if agent.id not in self.done_agents
        }
        rewards = {
            agent.id: self.sim.get_reward(agent.id) for agent in self.agents.values()
            if agent.id not in self.done_agents
        }
        dones = {
            agent.id: self.sim.get_done(agent.id) for agent in self.agents.values()
            if agent.id not in self.done_agents
        }
        infos = {
            agent.id: self.sim.get_info(agent.id) for agent in self.agents.values()
            if agent.id not in self.done_agents
        }

        for agent, done in dones.items():
            if done:
                self.done_agents.add(agent)

        # if all agents are done or the simulation is done, then return done
        if self.sim.get_all_done() or not (self.agents.keys() - self.done_agents):
            dones['__all__'] = True
        else:
            dones['__all__'] = False

        return obs, rewards, dones, infos
