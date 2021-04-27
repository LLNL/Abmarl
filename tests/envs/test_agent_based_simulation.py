
import pytest

from admiral.envs import AgentBasedSimulation, Agent, ActingAgent, ObservingAgent

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

def test_acting_agent_action_space():
    with pytest.raises(AssertionError):
        ActingAgent(id='agent', action_space=13)
    
    with pytest.raises(AssertionError):
        agent = ActingAgent(id='agent', action_space={'key': 'value'})
    
    agent = ActingAgent(id='agent')
    assert not agent.configured
    
    from gym.spaces import Discrete
    agent = ActingAgent(id='agent', action_space={'key': Discrete(12)})
    assert not agent.configured
    agent.finalize()
    assert agent.configured

def test_acting_agent_seed():
    from gym.spaces import Discrete
    agent = ActingAgent(id='agent', seed=24, action_space={
        1: Discrete(12),
        2: Discrete(3),
    })
    agent.finalize()
    assert agent.configured
    assert agent.action_space.sample() == {1: 6, 2: 2}

def test_observing_agent_action_space():
    with pytest.raises(AssertionError):
        ObservingAgent(id='agent', observation_space=13)
    
    with pytest.raises(AssertionError):
        agent = ObservingAgent(id='agent', observation_space={'key': 'value'})
    
    agent = ObservingAgent(id='agent')
    assert not agent.configured
    
    from gym.spaces import Discrete
    agent = ObservingAgent(id='agent', observation_space={'key': Discrete(12)})
    assert not agent.configured
    agent.finalize()
    assert agent.configured

# TODO: test a combination of acting and observing agents, esp with seeded action
    

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

