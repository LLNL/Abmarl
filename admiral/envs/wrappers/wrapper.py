from admiral.envs import AgentBasedSimulation


class Wrapper(AgentBasedSimulation):
    """
    Abstract Wrapper class implements the AgentBasedSimulation interface. The environment
    is stored and the environment agents are deep-copied. The interface functions
    calls are forwarded to the environment.
    """
    def __init__(self, env):
        """
        Wrap the environment and copy the agents.
        """
        assert isinstance(env, AgentBasedSimulation)
        self.env = env

        import copy
        self.agents = copy.deepcopy(env.agents)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

    def step(self, action, **kwargs):
        self.env.step(action, **kwargs)

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def get_obs(self, agent_id, **kwargs):
        return self.env.get_obs(agent_id, **kwargs)

    def get_reward(self, agent_id, **kwargs):
        return self.env.get_reward(agent_id, **kwargs)

    def get_done(self, agent_id, **kwargs):
        return self.env.get_done(agent_id, **kwargs)

    def get_all_done(self, **kwargs):
        return self.env.get_all_done(**kwargs)

    def get_info(self, agent_id, **kwargs):
        return self.env.get_info(agent_id, **kwargs)

    @property
    def unwrapped(self):
        """
        Fall through all the wrappers and obtain the original, completely unwrapped environment.
        """
        try:
            return self.env.unwrapped
        except AttributeError:
            return self.env
