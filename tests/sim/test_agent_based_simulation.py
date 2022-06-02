
import pytest

from abmarl.sim import AgentBasedSimulation, PrincipleAgent, ActingAgent, ObservingAgent, Agent, \
    DynamicOrderSimulation


def test_principle_agent_id():
    with pytest.raises(AssertionError):
        PrincipleAgent()

    with pytest.raises(AssertionError):
        PrincipleAgent(id=1)

    agent = PrincipleAgent(id='my_id')
    assert agent.id == 'my_id'
    assert agent.seed is None
    assert agent.configured

    with pytest.raises(AssertionError):
        agent.id = 4


def test_principle_agent_seed():
    with pytest.raises(AssertionError):
        PrincipleAgent(id='my_id', seed=13.5)
    agent = PrincipleAgent(id='my_id', seed=12)
    assert agent.seed == 12

    with pytest.raises(AssertionError):
        agent.seed = '12'


def test_principle_agents_equal():
    agent_1 = PrincipleAgent(id='1', seed=13)
    agent_2 = PrincipleAgent(id='1', seed=13)
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
    agent = ActingAgent(id='agent', seed=17, action_space={
        1: Discrete(12),
        2: Discrete(3),
    })
    agent.finalize()
    assert agent.configured
    assert agent.action_space.sample() == {1: 5, 2: 1}


def test_observing_agent_observation_space():
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


def test_agent():
    from gym.spaces import Discrete
    agent = Agent(
        id='agent', seed=7, observation_space={'obs': Discrete(2)},
        action_space={'act': Discrete(5)}
    )
    assert not agent.configured
    agent.finalize()
    assert agent.configured

    assert agent.action_space.sample() == {'act': 2}
    assert agent.observation_space.sample() == {'obs': 0}


def test_agent_based_simulation_agents():
    class ABS(AgentBasedSimulation):
        def __init__(self, agents):
            self.agents = agents

        def reset(self, **kwargs):
            pass

        def step(self, action, **kwargs):
            pass

        def render(self, **kwargs):
            pass

        def get_obs(self, agent_id, **kwargs):
            pass

        def get_reward(self, agent_id, **kwargs):
            pass

        def get_done(self, agent_id, **kwargs):
            pass

        def get_all_done(self, **kwargs):
            pass

        def get_info(self, agent_id, **kwargs):
            pass

    agents_single_object = PrincipleAgent(id='just_a_simple_agent')
    agents_list = [PrincipleAgent(id=f'{i}') for i in range(3)]
    agents_dict_key_id_no_match = {f'{i-1}': PrincipleAgent(id=f'{i}') for i in range(3)}
    agents_dict_bad_values = {f'{i}': 'PrincipleAgent(id=f"i")' for i in range(3)}
    agents_dict = {f'{i}': PrincipleAgent(id=f'{i}') for i in range(3)}

    with pytest.raises(AssertionError):
        ABS(agents=agents_single_object)

    with pytest.raises(AssertionError):
        ABS(agents=agents_list)

    with pytest.raises(AssertionError):
        ABS(agents=agents_dict_key_id_no_match)

    with pytest.raises(AssertionError):
        ABS(agents=agents_dict_bad_values)

    sim = ABS(agents=agents_dict)
    assert sim.agents == agents_dict
    sim.finalize()

    with pytest.raises(AssertionError):
        sim.agents = agents_single_object

    with pytest.raises(AssertionError):
        sim.agents = agents_list

    with pytest.raises(AssertionError):
        sim.agents = agents_dict_key_id_no_match

    with pytest.raises(AssertionError):
        ABS(agents=agents_dict_bad_values)


def test_dynamic_order_simulation():
    class SequentiallyFinishingSim(DynamicOrderSimulation):
        def __init__(self, **kwargs):
            self.agents = {f'agent{i}': PrincipleAgent(id=f'agent{i}') for i in range(4)}

        def reset(self, **kwargs):
            pass

        def step(self, action_dict, **kwargs):
            pass

        def render(self, **kwargs):
            pass

        def get_obs(self, agent_id, **kwargs):
            return {}

        def get_reward(self, agent_id, **kwargs):
            return {}

        def get_done(self, agent_id, **kwargs):
            return {}

        def get_all_done(self, **kwargs):
            return {}

        def get_info(self, agent_id, **kwargs):
            return {}

    sim = SequentiallyFinishingSim()
    sim.next_agent = 'agent0'
    assert sim.next_agent == ['agent0']
    sim.next_agent = ['agent1', 'agent2']
    assert sim.next_agent == ['agent1', 'agent2']
    sim.next_agent = ('agent3',)
    assert sim.next_agent == ('agent3',)
    sim.next_agent = set(('agent0', 'agent1'))
    assert sim.next_agent == set(('agent0', 'agent1'))

    # Expected to fail
    with pytest.raises(AssertionError):
        sim.next_agent = 3
    with pytest.raises(AssertionError):
        sim.next_agent = 'Agent4'
    with pytest.raises(AssertionError):
        sim.next_agent = ['agent0', 'agents1']
