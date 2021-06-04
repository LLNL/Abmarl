from ray.rllib import MultiAgentEnv


class MultiAgentWrapper(MultiAgentEnv):
    """
    Enable connection between SimulationManager and RLlib Trainer.

    Wraps a SimulationManager and forwards all calls to the manager. This class
    is boilerplate and needed because RLlib checks that the simulation is an instance
    of MultiAgentEnv.

    Attributes:
        env: The SimulationManager.
    """
    def __init__(self, env):
        from admiral.managers import SimulationManager
        assert isinstance(env, SimulationManager)
        self.env = env

    def reset(self):
        """See SimulationManager."""
        return self.env.reset()

    def step(self, actions):
        """See SimulationManager."""
        return self.env.step(actions)

    def render(self, *args, **kwargs):
        """See SimulationManager."""
        return self.env.render(*args, **kwargs)

    @property
    def unwrapped(self):
        """
        Fall through all the wrappers to the SimulationManager.

        Returns:
            The wrapped SimulationManager.
        """
        try:
            return self.env.unwrapped
        except AttributeError:
            return self.env
