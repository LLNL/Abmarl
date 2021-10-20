from itertools import cycle

from abmarl.sim import ActingAgent, ObservingAgent

from .simulation_manager import SimulationManager


class TurnBasedManager(SimulationManager):
    """
    The TurnBasedManager allows agents to take turns. The order of the agents is stored and the
    obs of the first agent is returned at reset. Each step returns the info of
    the next agent "in line". Agents who are done are removed from this line.
    Once all the agents are done, the manager returns all done.
    """
    def __init__(self, sim):
        super().__init__(sim)
        self.agent_order = cycle({
            agent_id: agent for agent_id, agent in self.agents.items()
            if (isinstance(agent, ActingAgent) and isinstance(agent, ObservingAgent))
        })

    def reset(self, **kwargs):
        """
        Reset the simulation and return the observation of the first agent.
        """
        self.done_agents = set()

        self.sim.reset(**kwargs)
        next_agent = next(self.agent_order)
        return {next_agent: self.sim.get_obs(next_agent)}

    def step(self, action_dict, **kwargs):
        """
        Assert that the incoming action does not come from an agent who is recorded
        as done. Step the simulation forward and return the observation, reward,
        done, and info of the next agent. If that next agent finished in this turn,
        then include the obs for the following agent, and so on until an agent
        is found that is not done. If all agents are done in this turn, then the
        wrapper returns all done.
        """
        agent_id = next(iter(action_dict))
        assert agent_id not in self.done_agents, \
            "Received an action for an agent that is already done."
        self.sim.step(action_dict, **kwargs)

        obs, rewards, dones, infos = {}, {}, {'__all__': self.sim.get_all_done()}, {}
        if dones['__all__']: # The simulation is done. Get output for all non-done agents
            for agent in self.agents:
                if agent in self.done_agents:
                    continue
                else:
                    obs[agent] = self.sim.get_obs(agent)
                    rewards[agent] = self.sim.get_reward(agent)
                    dones[agent] = self.sim.get_done(agent)
                    infos[agent] = self.sim.get_info(agent)
        else: # Simulation is not done. Get the output for the next agent(s).
            for next_agent in self.agent_order:
                # This agent was already done before, so there is no interaction
                # with it
                if next_agent in self.done_agents: continue

                # Check if the agent is just recently done:
                elif self.sim.get_done(next_agent):
                    # This agent only just recently finished. It sent an action before
                    # and now expects to receive an observation, rewrard, and done signal.
                    # So I want to add that to the output, but I don't want its action
                    # because it is done. So I want to include its info AND the info from
                    # the next not done agent.
                    obs[next_agent] = self.sim.get_obs(next_agent)
                    rewards[next_agent] = self.sim.get_reward(next_agent)
                    dones[next_agent] = self.sim.get_done(next_agent)
                    infos[next_agent] = self.sim.get_info(next_agent)
                    self.done_agents.add(next_agent)

                    # All agents could potentially be done now, so we check for that
                    if any([True for agent in self.agents if agent not in self.done_agents]):
                        continue
                    else:
                        # All agents are done
                        dones['__all__'] = True
                        break

                else:
                    # The agent is not done at all. So we grab its information and
                    # break the agent iteration loop
                    obs[next_agent] = self.sim.get_obs(next_agent)
                    rewards[next_agent] = self.sim.get_reward(next_agent)
                    dones[next_agent] = self.sim.get_done(next_agent)
                    infos[next_agent] = self.sim.get_info(next_agent)
                    break

        return obs, rewards, dones, infos
