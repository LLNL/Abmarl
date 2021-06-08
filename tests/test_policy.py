import numpy as np
import pytest

from abmarl.pols import GreedyPolicy, EpsilonSoftPolicy, RandomFirstActionPolicy


def test_abstract_policy():
    from abmarl.pols.policy import Policy
    with pytest.raises(TypeError):
        Policy(np.zeros((2,3)))


def test_greedy_policy_init():
    np.random.seed(24)
    table = np.random.normal(0, 1, size=(6,3))
    policy = GreedyPolicy(table)
    np.testing.assert_array_equal(table, policy.q_table)


def test_greedy_policy_act():
    np.random.seed(24)
    table = np.random.normal(0, 1, size=(6,3))
    policy = GreedyPolicy(table)
    action = policy.act(0)
    assert action == np.argmax(policy.q_table[0])


def test_greedy_policy_probability():
    np.random.seed(24)
    table = np.random.normal(0, 1, size=(6,3))
    policy = GreedyPolicy(table)
    prob = policy.probability(4, 2)
    assert prob == (1 if 2 == np.argmax(policy.q_table[4]) else 0)


def test_epsilon_soft_policy_init():
    np.random.seed(24)
    table = np.random.normal(0, 1, size=(6,3))
    policy = EpsilonSoftPolicy(table)
    np.testing.assert_array_equal(table, policy.q_table)
    assert policy.epsilon == 0.1


def test_epsilon_soft_policy_init_epsilon():
    np.random.seed(24)
    table = np.random.normal(0, 1, size=(6,3))
    with pytest.raises(AssertionError):
        EpsilonSoftPolicy(table, epsilon=-0.2)
    with pytest.raises(AssertionError):
        EpsilonSoftPolicy(table, epsilon=1.4)
    EpsilonSoftPolicy(table, epsilon=0.5)


def test_epsilon_soft_policy_act():
    np.random.seed(24)
    table = np.random.normal(0, 1, size=(6,20))
    policy = EpsilonSoftPolicy(table, epsilon=0.)
    action = policy.act(3)
    assert action == np.argmax(policy.q_table[3])


def test_epsilon_soft_policy_probability():
    np.random.seed(24)
    table = np.random.normal(0, 1, size=(6,3))
    policy = EpsilonSoftPolicy(table, epsilon=0.5)
    prob = policy.probability(4, 2)
    assert prob == (
        1 - policy.epsilon + policy.epsilon / policy.q_table[4].size
        if 2 == np.argmax(policy.q_table[4])
        else policy.epsilon / policy.q_table[4].size
    )


def test_random_first_action_policy_init():
    np.random.seed(24)
    table = np.random.normal(0, 1, size=(6,3))
    policy = RandomFirstActionPolicy(table)
    np.testing.assert_array_equal(table, policy.q_table)


def test_random_first_action_policy_reset():
    np.random.seed(24)
    table = np.random.normal(0, 1, size=(6,3))
    policy = RandomFirstActionPolicy(table)
    policy.reset()
    assert policy.take_random_action


def test_random_first_action_policy_no_reset():
    np.random.seed(24)
    table = np.random.normal(0, 1, size=(6,3))
    policy = RandomFirstActionPolicy(table)
    with pytest.raises(AttributeError):
        policy.act(5)
    with pytest.raises(AttributeError):
        policy.probability(1, 2)


def test_random_first_action_policy_act():
    np.random.seed(24)
    table = np.random.normal(0, 1, size=(6,3))
    policy = RandomFirstActionPolicy(table)

    policy.reset()
    policy.act(1)
    assert not policy.take_random_action
    action = policy.act(2)
    assert action == np.argmax(policy.q_table[2])


def test_random_first_action_policy_probability():
    np.random.seed(24)
    table = np.random.normal(0, 1, size=(6,3))
    policy = RandomFirstActionPolicy(table)

    policy.reset()
    prob = policy.probability(1, 1)
    assert prob == 1. / policy.q_table[1].size

    policy.act(2)
    prob = policy.probability(0, 2)
    assert prob == (1 if 2 == np.argmax(policy.q_table[4]) else 0)
