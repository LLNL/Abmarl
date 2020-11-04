
from gym import Env as GymEnv

class GymWrapper(GymEnv):
    """
    Wrap AgentEnvironment with only a single agent to the Gym Environment interface.
    This wrapper exposes the single agent's observation and action space directly
    in the environment.
    """
    def __init__(self, env):
        from admiral.managers import SimulationManager
        assert isinstance(env, SimulationManager)
        assert len(env.agents) == 1 # Can only work with single agents
        self.env = env
        self.agent_id, agent = next(iter(env.agents.items()))
        self.observation_space = agent.observation_space
        self.action_space = agent.action_space
    
    def reset(self, **kwargs):
        """
        Return the observation from the single agent.
        """
        obs = self.env.reset(**kwargs)
        return obs[self.agent_id]
    
    def step(self, action, **kwargs):
        """
        Wrap the action by storing it in a dict that maps the agent's id to the
        action. Pass to env.step. Return the observation, reward, done, and
        info from the single agent.
        """
        obs, reward, done, info = self.env.step({self.agent_id: action}, **kwargs)
        return obs[self.agent_id], \
            reward[self.agent_id], \
            done[self.agent_id], \
            info[self.agent_id]
    
    def render(self, **kwargs):
        """
        Forward render calls to the composed environment.
        """
        self.env.render(**kwargs)
    
    @property
    def unwrapped(self):
        """
        Fall through all the wrappers and obtain the original, completely unwrapped environment.
        """
        try:
            return self.env.unwrapped
        except:
            return self.env

