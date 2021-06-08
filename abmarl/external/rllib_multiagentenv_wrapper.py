from ray.rllib import MultiAgentEnv


class MultiAgentWrapper(MultiAgentEnv):
    """
    Enable connection between SimulationManager and RLlib Trainer.

    Wraps a SimulationManager and forwards all calls to the manager. This class
    is boilerplate and needed because RLlib checks that the simulation is an instance
    of MultiAgentEnv.

    Attributes:
        sim: The SimulationManager.
    """
    def __init__(self, sim):
        from abmarl.managers import SimulationManager
        assert isinstance(sim, SimulationManager)
        self.sim = sim

    def reset(self):
        """See SimulationManager."""
        return self.sim.reset()

    def step(self, actions):
        """See SimulationManager."""
        return self.sim.step(actions)

    def render(self, *args, **kwargs):
        """See SimulationManager."""
        return self.sim.render(*args, **kwargs)

    @property
    def unwrapped(self):
        """
        Fall through all the wrappers to the SimulationManager.

        Returns:
            The wrapped SimulationManager.
        """
        try:
            return self.sim.unwrapped
        except AttributeError:
            return self.sim
