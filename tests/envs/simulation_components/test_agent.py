
import numpy as np
import pytest

from admiral.envs.components import ComponentAgent

def test_component_agent_defaults():
    agent = ComponentAgent(id='agent')
    assert agent.initial_position is None
    np.testing.assert_array_equal(agent.min_max_health, np.array([0., 1.]))
    assert agent.initial_health is None
    assert agent.team == 0
    assert agent.is_alive
    assert agent.configured

def test_component_agent_initial_position():
    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', initial_position=[2, 4])
    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', initial_position=np.array([[2, 4]]))
    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', initial_position=np.array(['2', '4']))
    agent = ComponentAgent(id='agent', initial_position=np.array([2, 4]))
    np.testing.assert_array_equal(agent.initial_position, np.array([2, 4]))

def test_component_agent_min_max_health():
    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', min_max_health=[0, 1])
    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', min_max_health=np.array([[0, 1]]))
    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', min_max_health=np.array([1, 0]))
    agent = ComponentAgent(id='agent', min_max_health=np.array([2, 4]))
    np.testing.assert_array_equal(agent.min_max_health, np.array([2, 4]))

def test_component_agent_initial_health():
    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', min_max_health=np.array([0, 10]), initial_health='3')
    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', initial_health=2)
    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', initial_health=-2)
    agent = ComponentAgent(id='agent', initial_health=0.78)
    assert agent.initial_health == 0.78

def test_component_agent_team():
    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', team=2.0)
    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', team=0)
    agent = ComponentAgent(id='agent', team=2)
    assert agent.team == 2