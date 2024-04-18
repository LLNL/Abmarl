
from gymnasium import Env as GymEnv

from abmarl.sim import is_agent
from abmarl.sim import Agent, AgentBasedSimulation


class GymWrapper(GymEnv):
    """
    Wrap an AgentBasedSimulation object with only a single learning agent to the
    gym.Env interface. This wrapper exposes the single agent's observation and
    action space directly in the simulation.
    """
    def __init__(self, sim):
        from abmarl.managers import SimulationManager
        assert isinstance(sim, SimulationManager)
        learning_agents = {
            agent.id: agent for agent in sim.agents.values()
            if is_agent(agent)
        }
        assert len(learning_agents) == 1 # Can only work with single agents
        self.sim = sim
        self.agent_id, self.agent = next(iter(learning_agents.items()))

    @property
    def action_space(self):
        """
        The agent's action space is the environment's action space.
        """
        return self.agent.action_space

    @property
    def observation_space(self):
        """
        The agent's observation space is the environment's observation space.
        """
        return self.agent.observation_space

    @property
    def unwrapped(self):
        """
        Fall through all the wrappers and obtain the original, completely unwrapped simulation.
        """
        try:
            return self.sim.unwrapped
        except AttributeError:
            return self.sim

    def reset(self, **kwargs):
        """
        Return the observation from the single agent.
        """
        obs = self.sim.reset(**kwargs)
        return obs[self.agent_id], {}

    def step(self, action, **kwargs):
        """
        Wrap the action by storing it in a dict that maps the agent's id to the
        action. Pass to sim.step. Return the observation, reward, done, and
        info from the single agent.
        """
        obs, reward, done, info = self.sim.step({self.agent_id: action}, **kwargs)
        return obs[self.agent_id], \
            reward[self.agent_id], \
            done[self.agent_id], \
            False, \
            info[self.agent_id]

    def render(self, **kwargs):
        """
        Forward render calls to the composed simulation.
        """
        self.sim.render(**kwargs)


class GymABS(AgentBasedSimulation):
    """
    Wraps a GymEnv and leverages it for implementing the ABS interface.
    """
    def __init__(self, gym_env, null_observation, null_action, **kwargs):
        assert isinstance(gym_env, GymEnv), "gym_env must be a GymEnv."
        self._gym_env = gym_env
        agents = {'agent': Agent(
                id='agent',
                observation_space=gym_env.observation_space,
                null_observation=null_observation,
                action_space=gym_env.action_space,
                null_action=null_action
            )
        }
        super().__init__(agents=agents, **kwargs)
        # ABS storage
        self._obs = None
        self._reward = None
        self._done = None
        self._info = None

    def reset(self, **kwargs):
        """
        Reset the simulation and store the observation and info.
        """
        self._obs, self._info = self._gym_env.reset()

    def step(self, action, *args, **kwargs):
        """
        Step the simulation and store the relevant data.
        """
        self._obs, self._reward, term, trunc, self._info = self._gym_env.step(
            action, *args, **kwargs
        )
        self._done = term or trunc

    def render(self, **kwargs):
        self._gym_env.render(**kwargs)

    def get_obs(self, *args, **kwargs):
        """
        Return the stored observation, either from reset or step, whichever was last called.
        """
        return self._obs

    def get_reward(self, *args, **kwargs):
        """
        Return the stored reward, either from reset or step, whichever was last called.
        """
        return self._reward
    
    def get_done(self, *args, **kwargs):
        """
        Return the stored done status, either from reset or step, whichever was last called.
        """
        return self._done
    
    def get_all_done(self, **kwargs):
        """
        Same thing as get done.
        """
        return self.get_done()
    
    def get_info(self, *args, **kwargs):
        """
        Return the stored info, either from reset or step, whichever was last called.
        """
        return self._info


def gym_to_abmarl(
        gym_env,
        null_observation=None,
        null_action=None,
        ):
    return GymABS(
        gym_env,
        null_observation,
        null_action
    )
