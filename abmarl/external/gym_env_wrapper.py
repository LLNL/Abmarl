from gym import Env as GymEnv

from abmarl.sim import Agent


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
            if isinstance(agent, Agent)
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

    def reset(self, **kwargs):
        """
        Return the observation from the single agent.
        """
        obs = self.sim.reset(**kwargs)
        return obs[self.agent_id]

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
            info[self.agent_id]

    def render(self, **kwargs):
        """
        Forward render calls to the composed simulation.
        """
        self.sim.render(**kwargs)

    @property
    def unwrapped(self):
        """
        Fall through all the wrappers and obtain the original, completely unwrapped simulation.
        """
        try:
            return self.sim.unwrapped
        except AttributeError:
            return self.sim
