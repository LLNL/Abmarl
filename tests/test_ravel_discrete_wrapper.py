from abmarl.sim.wrappers.ravel_discrete_wrapper import ravel, unravel
from abmarl.sim.wrappers import RavelDiscreteWrapper
from abmarl.sim import Agent

from gym.spaces import MultiDiscrete, Discrete, MultiBinary, Box, Dict, Tuple
import numpy as np
import pytest
# from .helpers import FillInHelper, MultiAgentGymSpacesSim
from abmarl.sim.agent_based_simulation import AgentBasedSimulation

class FillInHelper(AgentBasedSimulation):
    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
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

class MultiAgentSim(FillInHelper):
    def __init__(self, num_agents=3):
        self.agents = {
            'agent' + str(i): Agent(
                id='agent'+str(i), observation_space=Discrete(2), action_space=Discrete(2)
            ) for i in range(num_agents)
        }

    def reset(self):
        self.action = {agent.id: None for agent in self.agents.values()}

    def step(self, action_dict):
        for agent_id, action in action_dict.items():
            self.action[agent_id] = action

    def get_obs(self, agent_id, **kwargs):
        return "Obs from " + agent_id

    def get_reward(self, agent_id, **kwargs):
        return "Reward from " + agent_id

    def get_done(self, agent_id, **kwargs):
        return "Done from " + agent_id

    def get_all_done(self, **kwargs):
        return "Done from all agents and/or simulation."

    def get_info(self, agent_id, **kwargs):
        return {'Action from ' + agent_id: self.action[agent_id]}


class MultiAgentGymSpacesSim(MultiAgentSim):
    def __init__(self):
        self.params = {'params': "there are none"}
        self.agents = {
            'agent0': Agent(
                id='agent0',
                observation_space=MultiBinary(4),
                action_space=Tuple((
                    Dict({
                        'first': Discrete(4),
                        'second': Box(low=-1, high=3, shape=(2,), dtype=int)
                    }),
                    MultiBinary(3)
                ))
            ),
            'agent1': Agent(
                id='agent1',
                observation_space=Box(low=0, high=1, shape=(1,), dtype=int),
                action_space=MultiDiscrete([4, 6, 2])
            ),
            'agent2': Agent(
                id='agent2',
                observation_space=MultiDiscrete([2, 2]),
                action_space=Dict({'alpha': MultiBinary(3)})
            ),
            'agent3': Agent(
                id='agent3',
                observation_space=Dict({
                    'first': Discrete(4),
                    'second': Box(low=-1, high=3, shape=(2,), dtype=int)
                }),
                action_space=Tuple((Discrete(3), MultiDiscrete([10, 10]), Discrete(2)))
            )
        }


    def get_obs(self, agent_id, **kwargs):
        if agent_id == 'agent0':
            return [0, 0, 0, 1]
        elif agent_id == 'agent1':
            return 0
        elif agent_id == 'agent2':
            return [1, 0]
        elif agent_id == 'agent3':
            return {'first': 1, 'second': [3, 1]}


    def get_info(self, agent_id, **kwargs):
        return self.action[agent_id]



def test_ravel():
    my_space = Dict({
        'a': MultiDiscrete([5, 3]),
        'b': MultiBinary(4),
        'c': Box(np.array([[-2, 6, 3],[0, 0, 1]]), np.array([[2, 12, 5],[2, 4, 2]]), dtype=int),
        'd': Dict({
            1: Discrete(3),
            2: Box(1, 3, (2,), int)
        }),
        'e': Tuple((
            MultiDiscrete([4, 1, 5]),
            MultiBinary(2),
            Dict({
                'my_dict': Discrete(11)
            })
        )),
        'f': Discrete(6),
    })
    point = {
        'a': [3, 1],
        'b': [0, 1, 1, 0],
        'c': np.array([[0, 7, 5],[1, 3, 1]]),
        'd': {1: 2, 2: np.array([1, 3])},
        'e': ([1,0,4], [1, 1], {'my_dict': 5}),
        'f': 1
    }
    ravelled_point = ravel(my_space, point)
    unravelled_point = unravel(my_space, ravelled_point)

    assert ravelled_point == 74748022765
    np.testing.assert_array_equal(unravelled_point['a'], point['a'])
    np.testing.assert_array_equal(unravelled_point['b'], point['b'])
    np.testing.assert_array_equal(unravelled_point['c'], point['c'])
    assert unravelled_point['d'][1] == point['d'][1]
    np.testing.assert_array_equal(unravelled_point['d'][2], point['d'][2])
    np.testing.assert_array_equal(unravelled_point['e'][0], point['e'][0])
    np.testing.assert_array_equal(unravelled_point['e'][1], point['e'][1])
    np.testing.assert_array_equal(unravelled_point['e'][2], point['e'][2])
    assert unravelled_point['f'] == point['f']


# Observations that we don't support
class FloatObservation(FillInHelper):
    def __init__(self):
        self.agents = {'agent0': Agent(
            id='agent0', observation_space=Box(-1.0, 1.0, shape=(4,)), action_space=Discrete(3)
        )}


class UnboundedBelowObservation(FillInHelper):
    def __init__(self):
        self.agents = {'agent0': Agent(
            id='agent0', observation_space=Box(
                np.array([0, 13, -3, -np.inf]),
                np.array([0, 20, 0, 0]),
                dtype=int
            ),
            action_space=Discrete(3)
        )}


class UnboundedAboveObservation(FillInHelper):
    def __init__(self):
        self.agents = {'agent0': Agent(
            id='agent0', observation_space=Box(
                np.array([0, 12, 20, 0]),
                np.array([np.inf, 20, 24, np.inf]),
                dtype=int
            ),
            action_space=Discrete(2)
        )}


# Actions that we don't support
class FloatAction(FillInHelper):
    def __init__(self):
        self.agents = {'agent0': Agent(
            id='agent0', observation_space=Box(-1.0, 1.0, shape=(4,)), action_space=Discrete(3)
        )}


class UnboundedBelowAction(FillInHelper):
    def __init__(self):
        self.agents = {'agent0': Agent(
            id='agent0',
            observation_space=Box(
                np.array([0, 13, -3, -np.inf]),
                np.array([0, 20, 0, 0]),
                dtype=int
            ),
            action_space=Discrete(3)
        )}


class UnboundedAboveAction(FillInHelper):
    def __init__(self):
        self.agents = {'agent0': Agent(
            id='agent0',
            observation_space=Box(
                np.array([0, 12, 20, 0]),
                np.array([np.inf, 20, 24, np.inf]),
                dtype=int
            ),
            action_space=Discrete(2)
        )}


def test_exceptions():
    with pytest.raises(AssertionError):
        RavelDiscreteWrapper(FloatObservation())
    with pytest.raises(AssertionError):
        RavelDiscreteWrapper(UnboundedBelowObservation())
    with pytest.raises(AssertionError):
        RavelDiscreteWrapper(UnboundedAboveObservation())
    with pytest.raises(AssertionError):
        RavelDiscreteWrapper(FloatAction())
    with pytest.raises(AssertionError):
        RavelDiscreteWrapper(UnboundedBelowAction())
    with pytest.raises(AssertionError):
        RavelDiscreteWrapper(UnboundedAboveAction())

def test_ravel_wrapper():
    sim = MultiAgentGymSpacesSim()
    wrapped_sim = RavelDiscreteWrapper(sim)
    assert wrapped_sim.unwrapped == sim
    for agent_id in wrapped_sim.agents:
        assert isinstance(wrapped_sim.agents[agent_id].observation_space, Discrete)
        assert isinstance(wrapped_sim.agents[agent_id].action_space, Discrete)
    sim = wrapped_sim

    sim.reset()
    assert sim.get_obs('agent0') == 1
    assert sim.get_obs('agent1') == 0
    assert sim.get_obs('agent2') == 2
    assert sim.get_obs('agent3') == 47

    action_0 = {
        'agent0': ({'first': 2, 'second': [-1, 2]}, [0, 1, 1]),
        'agent1': [3, 2, 0],
        'agent2': {'alpha': [1, 1, 0]},
        'agent3': (2, np.array([0, 6]), 1)
    }
    action_0_wrapped = {
        agent_id: sim.unwrap_action(sim.sim.agents[agent_id], action)
        for agent_id, action in action_0.items()
    }

    action_1 = {
        'agent0': ({'first': 0, 'second': [3, 3]}, [1, 1, 1]),
        'agent1': [1, 5, 1],
        'agent2': {'alpha': [1, 0, 0]},
        'agent3': (1, np.array([9, 4]), 0)
    }
    action_1_wrapped = {
        agent_id: sim.unwrap_action(sim.sim.agents[agent_id], action)
        for agent_id, action in action_1.items()
    }

    action_2 = {
        'agent0': ({'first': 1, 'second': [1, 0]}, [0, 0, 1]),
        'agent1': [2, 0, 1],
        'agent2': {'alpha': [0, 0, 0]},
        'agent3': (0, np.array([7, 7]), 0)
    }
    action_2_wrapped = {
        agent_id: sim.unwrap_action(sim.sim.agents[agent_id], action)
        for agent_id, action in action_2.items()
    }

    sim.step(action_0_wrapped)
    assert sim.get_obs('agent0') == 1
    assert sim.get_obs('agent1') == 0
    assert sim.get_obs('agent2') == 2
    assert sim.get_obs('agent3') == 47

    assert sim.get_reward('agent0') == 'Reward from agent0'
    assert sim.get_reward('agent1') == 'Reward from agent1'
    assert sim.get_reward('agent2') == 'Reward from agent2'
    assert sim.get_reward('agent3') == 'Reward from agent3'

    assert sim.get_done('agent0') == 'Done from agent0'
    assert sim.get_done('agent1') == 'Done from agent1'
    assert sim.get_done('agent2') == 'Done from agent2'
    assert sim.get_done('agent3') == 'Done from agent3'

    assert sim.get_info('agent0')[0]['first'] == action_0['agent0'][0]['first']
    np.testing.assert_array_equal(
        sim.get_info('agent0')[0]['second'], action_0['agent0'][0]['second']
    )
    np.testing.assert_array_equal(sim.get_info('agent0')[1], action_0['agent0'][1])
    np.testing.assert_array_equal(sim.get_info('agent1'), action_0['agent1'])
    np.testing.assert_array_equal(sim.get_info('agent2')['alpha'], action_0['agent2']['alpha'])
    np.testing.assert_array_equal(sim.get_info('agent3')[0], action_0['agent3'][0])
    np.testing.assert_array_equal(sim.get_info('agent3')[1], action_0['agent3'][1])
    np.testing.assert_array_equal(sim.get_info('agent3')[2], action_0['agent3'][2])

    sim.step(action_1_wrapped)
    assert sim.get_info('agent0')[0]['first'] == action_1['agent0'][0]['first']
    np.testing.assert_array_equal(
        sim.get_info('agent0')[0]['second'], action_1['agent0'][0]['second']
    )
    np.testing.assert_array_equal(sim.get_info('agent0')[1], action_1['agent0'][1])
    np.testing.assert_array_equal(sim.get_info('agent1'), action_1['agent1'])
    np.testing.assert_array_equal(sim.get_info('agent2')['alpha'], action_1['agent2']['alpha'])
    np.testing.assert_array_equal(sim.get_info('agent3')[0], action_1['agent3'][0])
    np.testing.assert_array_equal(sim.get_info('agent3')[1], action_1['agent3'][1])
    np.testing.assert_array_equal(sim.get_info('agent3')[2], action_1['agent3'][2])

    sim.step(action_2_wrapped)
    assert sim.get_info('agent0')[0]['first'] == action_2['agent0'][0]['first']
    np.testing.assert_array_equal(
        sim.get_info('agent0')[0]['second'], action_2['agent0'][0]['second']
    )
    np.testing.assert_array_equal(sim.get_info('agent0')[1], action_2['agent0'][1])
    np.testing.assert_array_equal(sim.get_info('agent1'), action_2['agent1'])
    np.testing.assert_array_equal(sim.get_info('agent2')['alpha'], action_2['agent2']['alpha'])
    np.testing.assert_array_equal(sim.get_info('agent3')[0], action_2['agent3'][0])
    np.testing.assert_array_equal(sim.get_info('agent3')[1], action_2['agent3'][1])
    np.testing.assert_array_equal(sim.get_info('agent3')[2], action_2['agent3'][2])
