
from ray.rllib import MultiAgentEnv


class MultiAgentWrapper(MultiAgentEnv):
    """
    Use this wrapper to activate multi-agent features on environments that do not directly inherit
    from ray.rllib.MultiAgentEnv.

    Environment must implement the following interface:
        reset -> observations: dict
        step(action: dict) -> observations: dict, rewards: dict, dones:dict, and infos:dict

    For rendering, the environment must implement the render function:
        render(fig: Matplotlib Artist)
    """
    def __init__(self, env):
        """
        env is an object of the environment class you want to wrap.
        """
        from admiral.managers import SimulationManager
        assert isinstance(env, SimulationManager)
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def render(self, *args, **kwargs):
        """
        If you want to vizualize agents in the environment, then your environment will need to
        implement the render function.
        """
        return self.env.render(*args, **kwargs)

    @property
    def unwrapped(self):
        """
        Fall through all the wrappers and obtain the original, completely unwrapped environment.
        """
        try:
            return self.env.unwrapped
        except AttributeError: # TODO: Confirm that this is really attribute error
            return self.env
