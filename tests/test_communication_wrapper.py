from gym.spaces import Discrete
import numpy as np

from abmarl.sim.wrappers import CommunicationHandshakeWrapper

from .helpers import MultiAgentGymSpacesSim


class CommsSimulation(MultiAgentGymSpacesSim):
    def get_obs(self, my_id, fusion_matrix={}):
        my_obs = super().get_obs(my_id)

        if 'agent1' in fusion_matrix and fusion_matrix['agent1']:
            agent_1_obs = self.get_obs('agent1')
            if my_id == 'agent0':
                my_obs[0] += agent_1_obs
            elif my_id == 'agent3':
                my_obs['first'] = agent_1_obs
        elif 'agent2' in fusion_matrix and fusion_matrix['agent2']:
            agent_2_obs = self.get_obs('agent2')
            if my_id == 'agent0':
                my_obs[1:2] += agent_2_obs[:]
            elif my_id == 'agent3':
                my_obs['second'] = agent_2_obs
        elif 'agent3' in fusion_matrix and fusion_matrix['agent3']:
            agent_3_obs = self.get_obs('agent3')
            if my_id == 'agent0':
                my_obs[0:-1] += [agent_3_obs['first'], agent_3_obs['second']][:]

        return my_obs


def test_communication_wrapper_init():
    sim = CommsSimulation()
    wrapped_sim = CommunicationHandshakeWrapper(sim)
    assert wrapped_sim.sim == sim
    assert wrapped_sim.agents != sim.agents
    assert wrapped_sim.unwrapped == sim

    for agent_id, agent in wrapped_sim.agents.items():
        assert agent.action_space['action'] == sim.agents[agent_id].action_space
        assert agent.observation_space['obs'] == sim.agents[agent_id].observation_space

        for other_agent in wrapped_sim.agents:
            if other_agent == agent_id: continue
            assert agent.action_space['send'][other_agent] == Discrete(2)
            assert agent.action_space['receive'][other_agent] == Discrete(2)
            assert agent.observation_space['message_buffer'][other_agent] == Discrete(2)


def test_communication_wrapper_reset():
    sim = CommunicationHandshakeWrapper(CommsSimulation())
    sim.reset()
    for values in sim.message_buffer.values():
        assert all([True if not val else False for val in values.values()])
    for values in sim.received_message.values():
        assert all([True if not val else False for val in values.values()])

    for agent_id in sim.agents:
        assert 'obs' in sim.get_obs(agent_id)
    assert sim.message_buffer == {
        agent_id: sim.get_obs(agent_id)['message_buffer'] for agent_id in sim.agents
    }


def test_communication_wrapper_step():
    sim = CommunicationHandshakeWrapper(CommsSimulation())
    sim.reset()

    action_0 = {
        'agent0': {
            'action': ({'first': 2, 'second': [-1, 2]}, [0, 1, 1]),
            'send': {'agent1': True, 'agent2': True, 'agent3': True},
            'receive': {'agent1': True, 'agent2': True, 'agent3': True},
        },
        'agent1': {
            'action': [3, 2, 0],
            'send': {'agent0': True, 'agent2': False, 'agent3': False},
            'receive': {'agent0': True, 'agent2': True, 'agent3': True},
        },
        'agent2': {
            'action': {'alpha': [1, 1, 0]},
            'send': {'agent0': True, 'agent1': True, 'agent3': False},
            'receive': {'agent0': True, 'agent1': True, 'agent3': True},
        },
        'agent3': {
            'action': (2, np.array([0, 6]), 1),
            'send': {'agent0': False, 'agent1': False, 'agent2': True},
            'receive': {'agent0': True, 'agent1': True, 'agent2': True},
        }
    }
    sim.step(action_0)
    for agent_id in sim.agents:
        agent_info = sim.get_info(agent_id)
        assert 'send' not in agent_info and 'receive' not in agent_info
    assert sim.get_obs('agent0')['obs'] == [0, 0, 0, 1]
    assert sim.get_obs('agent0')['message_buffer'] == {
        'agent1': True, 'agent2': True, 'agent3': False
    }
    assert sim.get_obs('agent1')['obs'] == 0
    assert sim.get_obs('agent1')['message_buffer'] == {
        'agent0': True, 'agent2': True, 'agent3': False
    }
    assert sim.get_obs('agent2')['obs'] == [1, 0]
    assert sim.get_obs('agent2')['message_buffer'] == {
        'agent0': True, 'agent1': False, 'agent3': True
    }
    assert sim.get_obs('agent3')['obs'] == {'first': 1, 'second': [3, 1]}
    assert sim.get_obs('agent3')['message_buffer'] == {
        'agent0': True, 'agent1': False, 'agent2': False
    }

    action_1 = {
        'agent0': {
            'action': ({'first': 0, 'second': [3, 3]}, [1, 1, 1]),
            'send': {'agent1': False, 'agent2': False, 'agent3': False},
            'receive': {'agent1': True, 'agent2': False, 'agent3': False},
        },
        'agent1': {
            'action': [1, 5, 1],
            'send': {'agent0': True, 'agent2': False, 'agent3': True},
            'receive': {'agent0': False, 'agent2': False, 'agent3': True},
        },
        'agent2': {
            'action': {'alpha': [1, 0, 0]},
            'send': {'agent0': True, 'agent1': True, 'agent3': False},
            'receive': {'agent0': True, 'agent1': True, 'agent3': False},
        },
        'agent3': {
            'action': (1, np.array([9, 4]), 0),
            'send': {'agent0': False, 'agent1': True, 'agent2': True},
            'receive': {'agent0': True, 'agent1': False, 'agent2': True},
        }
    }
    sim.step(action_1)
    assert sim.get_obs('agent0')['obs'] == [0, 0, 0, 1]
    assert sim.get_obs('agent0')['message_buffer'] == {
        'agent1': True, 'agent2': True, 'agent3': False
    }
    assert sim.get_obs('agent1')['obs'] == 0
    assert sim.get_obs('agent1')['message_buffer'] == {
        'agent0': False, 'agent2': True, 'agent3': True
    }
    assert sim.get_obs('agent2')['obs'] == [1, 0]
    assert sim.get_obs('agent2')['message_buffer'] == {
        'agent0': False, 'agent1': False, 'agent3': True
    }
    assert sim.get_obs('agent3')['obs'] == {'first': 1, 'second': [3, 1]}
    assert sim.get_obs('agent3')['message_buffer'] == {
        'agent0': False, 'agent1': True, 'agent2': False
    }

    action_2 = {
        'agent0': {
            'action': ({'first': 1, 'second': [1, 0]}, [0, 0, 1]),
            'send': {'agent1': True, 'agent2': True, 'agent3': True},
            'receive': {'agent1': False, 'agent2': True, 'agent3': True},
        },
        'agent1': {
            'action': [2, 0, 1],
            'send': {'agent0': True, 'agent2': False, 'agent3': False},
            'receive': {'agent0': True, 'agent2': True, 'agent3': True},
        },
        'agent2': {
            'action': {'alpha': [0, 0, 0]},
            'send': {'agent0': True, 'agent1': True, 'agent3': False},
            'receive': {'agent0': True, 'agent1': True, 'agent3': False},
        },
        'agent3': {
            'action': (0, np.array([7, 7]), 0),
            'send': {'agent0': False, 'agent1': False, 'agent2': True},
            'receive': {'agent0': True, 'agent1': True, 'agent2': False},
        }
    }
    sim.step(action_2)
    assert sim.get_obs('agent0')['obs'] == [0, 0, 1, 0, 0, 1]
    assert sim.get_obs('agent0')['message_buffer'] == {
        'agent1': True, 'agent2': True, 'agent3': False
    }
    assert sim.get_obs('agent1')['obs'] == 0
    assert sim.get_obs('agent1')['message_buffer'] == {
        'agent0': True, 'agent2': True, 'agent3': False
    }
    assert sim.get_obs('agent2')['obs'] == [1, 0]
    assert sim.get_obs('agent2')['message_buffer'] == {
        'agent0': True, 'agent1': False, 'agent3': True
    }
    assert sim.get_obs('agent3')['obs'] == {'first': 0, 'second': [3, 1]}
    assert sim.get_obs('agent3')['message_buffer'] == {
        'agent0': True, 'agent1': False, 'agent2': False
    }
