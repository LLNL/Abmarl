from abmarl.sim import AgentBasedSimulation


class Wrapper(AgentBasedSimulation):
    """
    Abstract Wrapper class implements the AgentBasedSimulation interface. The simulation
    is stored and the simulation agents are deep-copied. The interface functions
    calls are forwarded to the simulation.
    """
    def __init__(self, sim):
        """
        Wrap the simulation and copy the agents.
        """
        assert isinstance(sim, AgentBasedSimulation)
        self.sim = sim

        import copy
        self.agents = copy.deepcopy(sim.agents)

    def reset(self, **kwargs):
        self.sim.reset(**kwargs)

    def step(self, action, **kwargs):
        self.sim.step(action, **kwargs)

    def render(self, **kwargs):
        self.sim.render(**kwargs)

    def get_obs(self, agent_id, **kwargs):
        return self.sim.get_obs(agent_id, **kwargs)

    def get_reward(self, agent_id, **kwargs):
        return self.sim.get_reward(agent_id, **kwargs)

    def get_done(self, agent_id, **kwargs):
        return self.sim.get_done(agent_id, **kwargs)

    def get_all_done(self, **kwargs):
        return self.sim.get_all_done(**kwargs)

    def get_info(self, agent_id, **kwargs):
        return self.sim.get_info(agent_id, **kwargs)

    @property
    def unwrapped(self):
        """
        Fall through all the wrappers and obtain the original, completely unwrapped simulation.
        """
        try:
            return self.sim.unwrapped
        except AttributeError:
            return self.sim
