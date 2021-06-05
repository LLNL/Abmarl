from .wrapper import Wrapper


class SARWrapper(Wrapper):
    """
    Wraps the actions and observations for all the agents at reset and step.
    To create your own wrapper, inherit from this class and override the wrap
    and unwrap functions.

    Note: wrapping the action "goes the other way" than the reward and observation, like this:
        obs:    sim agent -> wrapper -> trainer
        reward: sim agent -> wrapper -> trainer
        action: sim agent <- wrapper <- trainer

    If you wrap an action, be aware that the wrapper must return what the simulation
    agents expect; whereas if you wrap an observation or reward, the wrapper must return
    what the trainer expects. The expectations are defined by the observation and
    action spaces of the wrapped simulation agents at initialization.
    """
    def step(self, action_dict, **kwargs):
        """
        Wrap each of the agent's actions from the policies before passing them
        to sim.step.
        """
        self.sim.step(
            {
                agent_id: self.wrap_action(self.sim.agents[agent_id], action)
                for agent_id, action in action_dict.items()
            },
            **kwargs
        )

    def get_obs(self, agent_id, **kwargs):
        return self.wrap_observation(self.sim.agents[agent_id], self.sim.get_obs(agent_id))

    def get_reward(self, agent_id, **kwargs):
        return self.wrap_reward(self.sim.get_reward(agent_id))

    # Default wrapping and unwrapping behavior. Override these in your custom wrapper.
    # Developer note: we have to have separate wrappers for each because we don't
    # want to force the observation and action space to map to the same wrapped space.
    def wrap_observation(self, from_agent, observation):
        return observation

    def unwrap_observation(self, from_agent, observation):
        return observation

    def wrap_action(self, from_agent, action):
        return action

    def unwrap_action(self, from_agent, action):
        return action

    def wrap_reward(self, reward):
        return reward

    def unwrap_reward(self, reward):
        return reward
