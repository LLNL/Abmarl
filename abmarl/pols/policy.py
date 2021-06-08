from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    """
    A policy maps a observation to an action. The relationship between the observations and the
    available actions is stored in a q_table. The act function chooses an action given a state.
    The probability function returns the probability of choosing an action given a state.
    """
    def __init__(self, q_table):
        """Store a q_table, which maps (state, action) to a value."""
        self.q_table = q_table

    @abstractmethod
    def act(self, state, *args, **kwargs):
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


class GreedyPolicy(Policy):
    """
    The GreedyPolicy will always choose the optimal action.
    """
    def act(self, state):
        return np.argmax(self.q_table[state])

    def probability(self, state, action):
        return 1 if action == np.argmax(self.q_table[state]) else 0


class EpsilonSoftPolicy(GreedyPolicy):
    """
    The EpsilonSoftPolicy will sample a uniform distribution between 0 and 1. If the sampled
    value is less than epsilon, then the policy will randomly choose an action. Otherwise, it
    will return the optimal action.
    """
    def __init__(self, *args, epsilon=0.1):
        super().__init__(*args)
        assert 0 <= epsilon <= 1.0
        self.epsilon = epsilon

    def act(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.q_table[state].size)
        else:
            return super().act(state)

    def probability(self, state, action):
        if action == np.argmax(self.q_table[state]): # Optimal action
            return 1 - self.epsilon + self.epsilon / self.q_table[state].size
        else: # Nonoptimal action
            return self.epsilon / self.q_table[state].size


class RandomFirstActionPolicy(GreedyPolicy):
    """
    The RandomFirstActionPolicy will choose a random action at the beginning of the episode.
    Afterwards, it will behave like a GreedyPolicy. Make sure you call the reset function at the
    beginning of every episode so that the policy knows to reset its parameters.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        """
        Set take_random_action to True so that the policy takes a random action at the beginning
        of an episode.
        """
        self.take_random_action = True

    def act(self, state):
        if self.take_random_action:
            action = np.random.randint(0, self.q_table[state].size)
        else:
            action = super().act(state)
        self.take_random_action = False
        return action

    def probability(self, state, action):
        if self.take_random_action:
            return 1. / self.q_table[state].size
        else:
            return super().probability(state, action)
