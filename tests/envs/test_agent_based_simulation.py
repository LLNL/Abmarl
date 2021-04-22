
import pytest

from admiral.envs import Agent, AgentBasedSimulation

def test_agent_id():
    with pytest.raises(AssertionError):
        Agent()
    
    with pytest.raises(AssertionError):
        Agent(id=1)

    agent = Agent(id='my_id')
    assert agent.id == 'my_id'
    assert agent.seed is None
    assert agent.configured

    with pytest.raises(AssertionError):
        agent.id = 4

def test_agent_seed():
    with pytest.raises(AssertionError):
        Agent(id='my_id', seed=13.5)
    agent = Agent(id='my_id', seed=12)
    assert agent.seed == 12

    with pytest.raises(AssertionError):
        agent.seed = '12'
    
def test_agents_equal():
    agent_1 = Agent(id='1', seed=13)
    agent_2 = Agent(id='1', seed=13)
    assert agent_1 == agent_2
    
    agent_2.id = '2'
    assert agent_1 != agent_2

    agent_2.id = '1'
    agent_2.seed = 12
    assert agent_1 != agent_2

