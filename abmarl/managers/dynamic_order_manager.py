
from abmarl.sim import DynamicOrderSimulation

from .simulation_manager import SimulationManager


class DynamicOrderManager(SimulationManager):
    """
    The DynamicOrderManager allows agents to take turns dynamically decided by the Simulation.

    The order of the agents is dynamically decided by the simulation as it runs.
    The simulation must be a DynamicOrderSimulation. The agents reported at reset
    and step are those given in the sim's next_agent property.
    """
    def __init__(self, sim):
        assert isinstance(sim, DynamicOrderSimulation), \
            "To use the DynamicOrderManager, the simulation must be a DynamicOrderSimulation."
        super().__init__(sim)

    def reset(self, **kwargs):
        """
        Reset the simulation and return the observation of the first agent.
        """
        self.done_agents = set()

        self.sim.reset(**kwargs)
        return {
            next_agent: self.sim.get_obs(next_agent) for next_agent in self.sim.next_agent
        }

    def step(self, action_dict, **kwargs):
        """
        Assert that the incoming action does not come from an agent who is recorded
        as done. Step the simulation forward and return the observation, reward,
        done, and info of the next agent. The simulation is responsible to ensure
        that there is at least one next_agent that did not finish in this turn,
        unless it is the last turn.
        """
        for agent_id in action_dict:
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
            for next_agent in self.sim.next_agent:
                # This agent was already done before, so there is no interaction
                # with it
                if next_agent in self.done_agents: continue

                # Check if the agent is just recently done:
                elif self.sim.get_done(next_agent):
                    # This agent only just recently finished. It sent an action before
                    # and now expects to receive an observation, reward, and done signal.
                    # The simulation is reponsible for providing multiple next_agents
                    # so that at least one of them is not done.
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
                    # The agent is not done at all, so we grab its information.
                    obs[next_agent] = self.sim.get_obs(next_agent)
                    rewards[next_agent] = self.sim.get_reward(next_agent)
                    dones[next_agent] = self.sim.get_done(next_agent)
                    infos[next_agent] = self.sim.get_info(next_agent)

        return obs, rewards, dones, infos
