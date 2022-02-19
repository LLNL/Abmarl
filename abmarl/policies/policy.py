from abc import ABC, abstractmethod

from gym.spaces import Space


class Policy(ABC):
    """
    A policy maps an observation to an action.

    The underlying mapping function can take many forms. It can be Q-table, a non
    linear function, a neural network, etc.
    """
    def __init__(self, action_space=None, observation_space=None, **kwargs):
        self.action_space = action_space
        self.observation_space = observation_space

    @property
    def action_space(self):
        """
        The space of possible output actions for this policy.
        """
        return self._action_space

    @action_space.setter
    def action_space(self, value):
        assert isinstance(value, Space), "Action space must be a gym space."
        self._action_space = value

    @property
    def observation_space(self):
        """
        The space of possible input observations for this policy.
        """
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        assert isinstance(value, Space), "Observation space must be a gym space."
        self._observation_space = value

    @abstractmethod
    def compute_action(self, obs, **kwargs):
        """
        Generate an action from the given observation.

        Args:
            obs: The input observation.

        Returns:
            An action computed by feeding the observation forward through the underlying
            mapping.
        """
        pass

    @abstractmethod
    def update(self, obs, action, value, **kwargs):
        """
        Update the policy with the data in obs, action, and value.

        Args:
            obs: The observation involved in the update.
            action: The action involved in the update.
            value: The value to use for the update.
        """
        pass

    def reset(self):
        """
        Reset the policy at the beginning of an episode.

        Some policies behave differently at the beginning of an episode or as an episode
        progresses. The reset function allows them to reset their parameters accordingly.
        """
        pass


class RandomPolicy(Policy):
    """
    Generate random actions.
    """
    def compute_action(self, obs, **kwargs):
        """
        Check that the observation is in the space, then return a random action.

        Args:
            obs: The input observation.

        Returns:
            A random action from the action space.
        """
        assert obs in self.observation_space
        return self.action_space.sample()

    def update(self, *args, **kwargs):
        """
        RandomPolicy does not chagne, so update does nothing.
        """
        pass
