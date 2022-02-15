
from abc import ABC, abstractmethod

from gym.spaces import Discrete
import numpy as np

from .policy import Policy

# TODO: Remove this class and let the subclasses, like greedy policy, work with
# different implementations of the Q insead of forcing it to be a table.
class QTablePolicy(Policy, ABC):
    """
    A policy that explicitly stores and updates a Q-table.

    This requires Discrete observation space and action space.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q_table = np.random.normal(0, 1, size=(self.observation_space.n, self.action_space.n))

    @property
    def q_table(self):
        """
        The Q-table stores the value of each state-action pair.

        The rows of the table should be the observations and the columns should
        be actions.
        """
        return self._q_table

    @q_table.setter
    def q_table(self, value):
        assert isinstance(value, np.ndarray), "Q-Table must be a numpy array."
        assert len(value.shape) == 2, "Q-Table must have two dimensions."
        assert value.shape[0] == self.observation_space.n, \
            "The number of rows in the q table must be the same as the size of " + \
            "the observation space."
        assert value.shape[1] == self.action_space.n, \
            "The number of columns in the q table must be the same as the size of " + \
            "the action space."
        self._q_table = value

    @property
    def action_space(self):
        """
        The space of possible output actions for this policy.

        Must be Discrete.
        """
        return self._action_space

    @action_space.setter
    def action_space(self, value):
        assert isinstance(value, Discrete), "Action space must be Discrete."
        self._action_space = value

    @property
    def observation_space(self):
        """
        The space of possible input observations for this policy.

        Must be Discrete.
        """
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        assert isinstance(value, Discrete), "Observation space must be Discrete."
        self._observation_space = value

    @classmethod
    def build(cls, policy_or_table, **kwargs):
        """
        Build a policy from another policy or q_table.

        Args:
            policy_or_table: We can create a policy using another policy, in which
                case we look at the policy's q_table. Or we can create a policy
                from the q_table directly.

        Returns:
            A new policy with the q_table.
        """
        if isinstance(policy_or_table, QTablePolicy):
            q_table = policy_or_table.q_table
        else:
            q_table = policy_or_table
        new_policy = cls(
            observation_space=Discrete(q_table.shape[0]),
            action_space=Discrete(q_table.shape[1]),
            **kwargs
        )
        new_policy.q_table = q_table
        return new_policy

    @abstractmethod
    def probability(self, obs, action, **kwargs):
        """
        Calculate the probability of choosing this action given this observation.
        """
        pass

class GreedyPolicy(QTablePolicy):
    """
    The GreedyPolicy will always choose the optimal action.
    """
    def compute_action(self, obs, **kwargs):
        return np.argmax(self.q_table[obs])

    def probability(self, obs, action, **kwargs):
        return 1 if action == np.argmax(self.q_table[obs]) else 0


class EpsilonSoftPolicy(GreedyPolicy):
    """
    Choose random action with some probability.

    The EpsilonSoftPolicy will sample a uniform distribution between 0 and 1. If the sampled
    value is less than epsilon, then the policy will randomly choose an action. Otherwise, it
    will return the optimal action.
    """
    def __init__(self, epsilon=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    @property
    def epsilon(self):
        """
        The probability of choosing a random action.
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        assert 0 <= value <= 1.0, "Epsilon must be between 0 and 1."
        self._epsilon = value

    def compute_action(self, obs, **kwargs):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.q_table[obs].size)
        else:
            return super().compute_action(obs, **kwargs)

    def probability(self, obs, action, **kwargs):
        if action == np.argmax(self.q_table[obs]): # Optimal action
            return 1 - self.epsilon + self.epsilon / self.q_table[obs].size
        else: # Nonoptimal action
            return self.epsilon / self.q_table[obs].size
