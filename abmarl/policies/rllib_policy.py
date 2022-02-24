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


class RandomAction(HeuristicPolicy):
    """A policy to take random actions."""
    def __init__(self, observation_space, action_space, config={}):
        super().__init__(observation_space, action_space, config)

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
