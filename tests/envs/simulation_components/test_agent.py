
import numpy as np
import pytest

from admiral.envs.components import ComponentAgent

def test_component_agent_initial_position():
    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', initial_position=[2, 4])

    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', initial_position=np.array([[2, 4]]))

    with pytest.raises(AssertionError):
        ComponentAgent(id='agent', initial_position=np.array(['2', '4']))

    agent = ComponentAgent(id='agent', initial_position=np.array([2, 4]))
    np.testing.assert_array_equal(agent.initial_position, np.array([2, 4]))

# def test_component_agent_min_max_health():
#     pass

# def test_component_agent_initial_health():
#     pass

# def test_component_agent_team():
#     pass
