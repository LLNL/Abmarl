
import numpy as np
import pytest

from admiral.component_envs.movement import GridMovementEnv
from admiral.component_envs.world import WorldAgent, GridWorldEnv

def test_agents_moving_in_grid():
    region = 10
    agents = {
        'agent0': WorldAgent(id='agent0', starting_position=np.array([6, 4])),
        'agent1': WorldAgent(id='agent1', starting_position=np.array([3, 3])),
        'agent2': WorldAgent(id='agent2', starting_position=np.array([0, 1])),
        'agent3': WorldAgent(id='agent3', starting_position=np.array([8, 4])),
    }
    env = GridMovementEnv(
        region=region,
        agents=agents
    )
    assert isinstance(env, GridWorldEnv)
    env.reset()
    for agent in env.agents.values():
        assert agent.position is not None

    assert env.process_move(env.agents['agent0'], np.array([-1, 1]))
    assert env.process_move(env.agents['agent1'], np.array([0, 1]))
    assert env.process_move(env.agents['agent2'], np.array([0, -1]))
    assert env.process_move(env.agents['agent3'], np.array([1, 0]))

    np.testing.assert_array_equal(env.agents['agent0'].position, np.array([5, 5]))
    np.testing.assert_array_equal(env.agents['agent1'].position, np.array([3, 4]))
    np.testing.assert_array_equal(env.agents['agent2'].position, np.array([0, 0]))
    np.testing.assert_array_equal(env.agents['agent3'].position, np.array([9, 4]))

    assert env.process_move(env.agents['agent0'], np.array([2, -2]))
    assert env.process_move(env.agents['agent1'], np.array([-3, 0]))
    assert not env.process_move(env.agents['agent2'], np.array([-1, 0]))
    assert not env.process_move(env.agents['agent3'], np.array([1, 1]))

    np.testing.assert_array_equal(env.agents['agent0'].position, np.array([7, 3]))
    np.testing.assert_array_equal(env.agents['agent1'].position, np.array([0, 4]))
    np.testing.assert_array_equal(env.agents['agent2'].position, np.array([0, 0]))
    np.testing.assert_array_equal(env.agents['agent3'].position, np.array([9, 4]))
