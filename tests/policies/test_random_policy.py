
from gym.spaces import Discrete
import pytest

from abmarl.policies.policy import RandomPolicy


def test_policy_action_space():
    with pytest.raises(AssertionError):
        RandomPolicy(observation_space=Discrete(3))
    with pytest.raises(AssertionError):
        RandomPolicy(observation_space=Discrete(3), action_space={'move': Discrete(3)})
    RandomPolicy(observation_space=Discrete(3), action_space=Discrete(9))


def test_policy_observation_space():
    with pytest.raises(AssertionError):
        RandomPolicy(action_space=Discrete(3))
    with pytest.raises(AssertionError):
        RandomPolicy(action_space=Discrete(3), observation_space={'move': Discrete(3)})
    RandomPolicy(action_space=Discrete(3), observation_space=Discrete(9))


def test_compute_action():
    policy = RandomPolicy(action_space=Discrete(4), observation_space=Discrete(2))
    with pytest.raises(AssertionError):
        policy.compute_action(3)
    action = policy.compute_action(0)
    assert action in policy.action_space
