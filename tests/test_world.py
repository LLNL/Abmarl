
import numpy as np
import pytest

from admiral.component_envs.world import GridWorldEnv
from admiral.component_envs.world import WorldAgent

def test_no_region():
    with pytest.raises(AssertionError):
        GridWorldEnv()

def test_just_region():
    env = GridWorldEnv(region=10)
    assert env.region == 10
    assert env.agents == {}

    env.reset()

def test_agents_with_starting_positions():
    region = 10
    agents = {
        'agent0': WorldAgent(id='agent0', starting_position=np.array([6, 4])),
        'agent1': WorldAgent(id='agent1', starting_position=np.array([3, 3])),
        'agent2': WorldAgent(id='agent2', starting_position=np.array([0, 1])),
        'agent3': WorldAgent(id='agent3', starting_position=np.array([8, 4])),
    }
    env = GridWorldEnv(
        region=region,
        agents=agents
    )
    assert env.region == region
    np.testing.assert_array_equal(env.agents['agent0'].starting_position, np.array([6, 4]))
    np.testing.assert_array_equal(env.agents['agent1'].starting_position, np.array([3, 3]))
    np.testing.assert_array_equal(env.agents['agent2'].starting_position, np.array([0, 1]))
    np.testing.assert_array_equal(env.agents['agent3'].starting_position, np.array([8, 4]))

    env.reset()
    for agent in env.agents.values():
        np.testing.assert_array_equal(agent.position, agent.starting_position)

def test_agents_with_random_positions():
    np.random.seed(24)
    env = GridWorldEnv(
        region=10,
        agents={f'agent{i}': WorldAgent(id=f'agent{i}') for i in range(4)}
    )
    assert env.agents['agent0'] == WorldAgent(id='agent0')
    assert env.agents['agent1'] == WorldAgent(id='agent1')
    assert env.agents['agent2'] == WorldAgent(id='agent2')
    assert env.agents['agent3'] == WorldAgent(id='agent3')
        
    env.reset()
    np.testing.assert_array_equal(env.agents['agent0'].position, np.array([2, 3]))
    np.testing.assert_array_equal(env.agents['agent1'].position, np.array([0, 7]))
    np.testing.assert_array_equal(env.agents['agent2'].position, np.array([1, 1]))
    np.testing.assert_array_equal(env.agents['agent3'].position, np.array([1, 4]))
