
import numpy as np
import pytest

from admiral.component_envs.movement import GridMovementEnv

def test_agents_moving_in_grid():
    env = GridMovementEnv(region=10)

    np.testing.assert_array_equal(env.process_move(np.array([6, 4]), np.array([-1, 1])), np.array([5, 5]))
    np.testing.assert_array_equal(env.process_move(np.array([3, 3]), np.array([0, 1])), np.array([3, 4]))
    np.testing.assert_array_equal(env.process_move(np.array([0, 1]), np.array([0, -1])), np.array([0, 0]))
    np.testing.assert_array_equal(env.process_move(np.array([8, 4]), np.array([1, 0])), np.array([9, 4]))

    np.testing.assert_array_equal(env.process_move(np.array([5, 5]), np.array([2, -2])), np.array([7, 3]))
    np.testing.assert_array_equal(env.process_move(np.array([3, 4]), np.array([-3, 0])), np.array([0, 4]))
    np.testing.assert_array_equal(env.process_move(np.array([0, 0]), np.array([-1, 0])), np.array([0, 0]))
    np.testing.assert_array_equal(env.process_move(np.array([9, 4]), np.array([1, 1])), np.array([9, 4]))
