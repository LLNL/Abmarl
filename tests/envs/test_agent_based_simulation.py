
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

def test_agent_based_simulation_agents():
    class ABS(AgentBasedSimulation):
        def __init__(self, agents):
            self.agents = agents
        
        def reset(self, **kwargs): pass
        def step(self, action, **kwargs): pass
        def render(self, **kwargs): pass    
        def get_obs(self, agent_id, **kwargs): pass    
        def get_reward(self, agent_id, **kwargs): pass    
        def get_done(self, agent_id, **kwargs): pass    
        def get_all_done(self, **kwargs): pass    
        def get_info(self, agent_id, **kwargs): pass

    agents_single_object = Agent(id='just_a_simple_agent')
    agents_list = [Agent(id=f'{i}') for i in range(3)]
    agents_dict_key_id_no_match = {f'{i-1}': Agent(id=f'{i}') for i in range(3)}
    agents_dict_bad_values = {f'{i}': 'Agent(id=f"i")' for i in range(3)}
    agents_dict = {f'{i}': Agent(id=f'{i}') for i in range(3)}
    
    with pytest.raises(AssertionError):
        ABS(agents=agents_single_object)
    
    with pytest.raises(AssertionError):
        ABS(agents=agents_list)
    
    with pytest.raises(AssertionError):
        ABS(agents=agents_dict_key_id_no_match)
    
    with pytest.raises(AssertionError):
        ABS(agents=agents_dict_bad_values)
    
    env = ABS(agents=agents_dict)
    assert env.agents == agents_dict
    env.finalize()
    
    with pytest.raises(AssertionError):
        env.agents = agents_single_object
    
    with pytest.raises(AssertionError):
        env.agents = agents_list
    
    with pytest.raises(AssertionError):
        env.agents = agents_dict_key_id_no_match
    
    with pytest.raises(AssertionError):
        ABS(agents=agents_dict_bad_values)

