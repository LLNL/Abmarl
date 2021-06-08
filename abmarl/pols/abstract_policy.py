from abc import ABC, abstractmethod

from ray.rllib.policy.policy import Policy


class HeuristicPolicy(Policy, ABC):
    """Abstract class for policies that do not learn."""
    @abstractmethod
    def compute_actions(self, *args, **kwargs):
        pass

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
