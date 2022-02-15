from abc import ABC, abstractmethod

from gym.spaces import Space, Discrete
import numpy as np


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
            policy.
        """
        pass

    def reset(self):
        """
        Reset the policy at the beginning of an episode.

        Some policies behave differently at the beginning of an episode or as an episode
        progresses. The reset function allows them to reset their parameters accordingly.
        """
        pass


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
            action_spce=Discrete(q_table.shape[1],
            **kwargs)
        )
        new_policy.q_table = q_table
        return new_policy

class GreedyPolicy(QTablePolicy):
    """
    The GreedyPolicy will always choose the optimal action.
    """
    def compute_action(self, obs, **kwargs):
        return np.argmax(self.q_table[obs])


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


class RandomFirstActionPolicy(GreedyPolicy):
    """
    First action is random, all other actions are greedy.

    The RandomFirstActionPolicy will choose a random action at the beginning of the episode.
    Afterwards, it will behave like a GreedyPolicy.
    """
    def reset(self):
        """
        Set take_random_action to True so that the policy takes a random action at the beginning
        of an episode.
        """
        self._take_random_action = True

    def compute_action(self, obs, **kwargs):
        if self._take_random_action:
            action = np.random.randint(0, self.q_table[obs].size)
        else:
            action = super().compute_action(obs, **kwargs)
        self._take_random_action = False
        return action


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


class _QPolicy(ABC):
    """
    A policy maps a observation to an action. The relationship between the observations and the
    available actions is stored in a q_table. The act function chooses an action given a state.
    The probability function returns the probability of choosing an action given a state.
    """
    def __init__(self, q_table):
        """Store a q_table, which maps (state, action) to a value."""
        self.q_table = q_table

    @abstractmethod
    def compute_action(self, state, *args, **kwargs):
        """Choose an action given a state."""
        pass

    @abstractmethod
    def probability(self, state, action):
        """Calculate the probability of choosing this action given this state."""
        pass

    def reset(self):
        """
        Some policies behave differently at the beginning of an episode or as an episode
        progresses. The reset function allows them to reset their parameters accordingly.
        """
        pass


class _GreedyPolicy(_QPolicy):
    """
    The GreedyPolicy will always choose the optimal action.
    """
    def compute_action(self, state):
        return np.argmax(self.q_table[state])

    def probability(self, state, action):
        return 1 if action == np.argmax(self.q_table[state]) else 0


class _EpsilonSoftPolicy(_GreedyPolicy):
    """
    The EpsilonSoftPolicy will sample a uniform distribution between 0 and 1. If the sampled
    value is less than epsilon, then the policy will randomly choose an action. Otherwise, it
    will return the optimal action.
    """
    def __init__(self, *args, epsilon=0.1):
        super().__init__(*args)
        assert 0 <= epsilon <= 1.0
        self.epsilon = epsilon

    def compute_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.q_table[state].size)
        else:
            return super().compute_action(state)

    def probability(self, state, action):
        if action == np.argmax(self.q_table[state]): # Optimal action
            return 1 - self.epsilon + self.epsilon / self.q_table[state].size
        else: # Nonoptimal action
            return self.epsilon / self.q_table[state].size
