
from gym.spaces import Discrete, MultiBinary
import numpy as np
import pytest

from abmarl.policies.q_table_policy import QTablePolicy, EpsilonSoftPolicy


class QPolicyTester(QTablePolicy):
    def reset(self):
        self.first_guess = True

    def compute_action(self, obs, **kwargs):
        pass

    def probability(self, obs, action, **kwargs):
        pass


def test_q_policy_init_and_properties():
    with pytest.raises(AssertionError):
        QPolicyTester(observation_space=Discrete(3), action_space=MultiBinary(4))
    with pytest.raises(AssertionError):
        QPolicyTester(observation_space=MultiBinary(2), action_space=Discrete(10))
    policy = QPolicyTester(observation_space=Discrete(3), action_space=(Discrete(10)))
    assert policy.q_table.shape == (3, 10)
    assert policy.action_space == Discrete(10)
    assert policy.observation_space == Discrete(3)
    policy = QPolicyTester(
        observation_space=Discrete(3),
        action_space=(Discrete(10)),
        q_table=np.random.randint(0, 5, (3, 10))
    )

    with pytest.raises(AssertionError):
        QPolicyTester(
            observation_space=Discrete(2),
            action_space=Discrete(10),
            q_table=np.random.normal(0, 1, (3, 10))
        )
    with pytest.raises(AssertionError):
        policy.q_table = Discrete(10)
    with pytest.raises(AssertionError):
        policy.action_space = MultiBinary(10)
    with pytest.raises(AssertionError):
        policy.observation_space = MultiBinary(10)


def test_q_policy_update():
    q = np.array([
        [ 0,  1,  2],
        [-1,  1, -2],
        [ 0,  1,  0],
        [ 2, -2, -2]])
    policy = QPolicyTester(
        observation_space=Discrete(q.shape[0]),
        action_space=Discrete(q.shape[1]),
        q_table=q
    )
    policy.update(0, 1, 3)
    policy.update(2, 2, -1)
    policy.update(3, 0, -2)
    np.testing.assert_array_equal(
        policy.q_table,
        np.array([
            [ 0,  3,  2],
            [-1,  1, -2],
            [ 0,  1, -1],
            [-2, -2, -2]
        ])
    )


def test_greedy_policy():
    q = np.array([
        [ 0,  1,  2],
        [-1,  1, -2],
        [ 0,  1,  0],
        [ 2, -2, -2]])
    policy = EpsilonSoftPolicy(
        observation_space=Discrete(q.shape[0]),
        action_space=Discrete(q.shape[1]),
        q_table=q,
        epsilon=0
    )
    assert policy.compute_action(0) == 2
    assert policy.probability(0, 0) == 0
    assert policy.probability(0, 1) == 0
    assert policy.probability(0, 2) == 1

    assert policy.compute_action(1) == 1
    assert policy.probability(1, 0) == 0
    assert policy.probability(1, 1) == 1
    assert policy.probability(1, 2) == 0

    assert policy.compute_action(2) == 1
    assert policy.probability(2, 0) == 0
    assert policy.probability(2, 1) == 1
    assert policy.probability(2, 2) == 0

    assert policy.compute_action(3) == 0
    assert policy.probability(3, 0) == 1
    assert policy.probability(3, 1) == 0
    assert policy.probability(3, 2) == 0


def test_epsilon_soft_init_and_build():
    policy = EpsilonSoftPolicy(
        observation_space=Discrete(3),
        action_space=Discrete(5),
        epsilon=0.6
    )
    assert policy.epsilon == 0.6

    policy = EpsilonSoftPolicy(
        observation_space=policy.observation_space,
        action_space=policy.action_space,
        q_table=policy._q_table,
        epsilon=0.2
    )
    assert policy.epsilon == 0.2

    with pytest.raises(AssertionError):
        EpsilonSoftPolicy(
            observation_space=Discrete(3),
            action_space=Discrete(5),
            epsilon=-0.6
        )

    with pytest.raises(AssertionError):
        EpsilonSoftPolicy(
            observation_space=policy.observation_space,
            action_space=policy.action_space,
            q_table=policy.q_table,
            epsilon=1.6
        )


def test_epsilon_soft_compute_action_and_probability():
    np.random.seed(24)
    q = np.array([
        [ 0,  1,  2],
        [-1,  1, -2],
        [ 0,  1,  0],
        [ 2, -2, -2]])
    policy = EpsilonSoftPolicy(
        observation_space=Discrete(q.shape[0]),
        action_space=Discrete(q.shape[1]),
        q_table=q,
        epsilon=0.5
    )
    assert policy.compute_action(0) == 2 # random
    assert policy.probability(0, 0) == 0.5 / 3
    assert policy.probability(0, 1) == 0.5 / 3
    assert policy.probability(0, 2) == 1 - 0.5 + 0.5 / 3

    assert policy.compute_action(1) == 1 # random
    assert policy.probability(1, 0) == 0.5 / 3
    assert policy.probability(1, 1) == 1 - 0.5 + 0.5 / 3
    assert policy.probability(1, 2) == 0.5 / 3

    assert policy.compute_action(2) == 1 # random
    assert policy.probability(2, 0) == 0.5 / 3
    assert policy.probability(2, 1) == 1 - 0.5 + 0.5 / 3
    assert policy.probability(2, 2) == 0.5 / 3

    assert policy.compute_action(3) == 0 # random
    assert policy.probability(3, 0) == 1 - 0.5 + 0.5 / 3
    assert policy.probability(3, 1) == 0.5 / 3
    assert policy.probability(3, 2) == 0.5 / 3
