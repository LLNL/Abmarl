
from gym.spaces import Discrete
import numpy as np

from admiral.envs.wrappers import CommunicationWrapper
from admiral.envs import Agent

from .helpers import MultiAgentGymSpacesEnv

class CommsEnv(MultiAgentGymSpacesEnv):
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
    env = CommsEnv()
    wrapped_env = CommunicationWrapper(env)
    assert wrapped_env.env == env
    assert wrapped_env.agents != env.agents
    assert wrapped_env.unwrapped == env

    for agent_id, agent in wrapped_env.agents.items():
        assert agent.action_space['env_action'] == env.agents[agent_id].action_space
        assert agent.observation_space['env_obs'] == env.agents[agent_id].observation_space

        for other_agent in wrapped_env.agents:
            if other_agent == agent_id: continue
            assert agent.action_space['send'][other_agent] == Discrete(2)
            assert agent.action_space['receive'][other_agent] == Discrete(2)
            assert agent.observation_space['message_buffer'][other_agent] == Discrete(2)

def test_communication_wrapper_reset():
    env = CommunicationWrapper(CommsEnv())
    env.reset()
    for values in env.message_buffer.values():
        assert all([True if val == False else False for val in values.values()])
    for values in env.received_message.values():
        assert all([True if val == False else False for val in values.values()])

    for agent_id in env.agents:
        assert 'env_obs' in env.get_obs(agent_id)
    assert env.message_buffer == {agent_id: env.get_obs(agent_id)['message_buffer'] for agent_id in env.agents}

def test_communication_wrapper_step():
    env = CommunicationWrapper(CommsEnv())
    env.reset()

    action_0 = {
        'agent0': {
            'env_action': ({'first': 2, 'second': [-1, 2]}, [0, 1, 1]),
            'send': {'agent1': True, 'agent2': True, 'agent3': True},
            'receive': {'agent1': True, 'agent2': True, 'agent3': True},
        },
        'agent1': {
            'env_action': [3, 2, 0],
            'send': {'agent0': True, 'agent2': False, 'agent3': False},
            'receive': {'agent0': True, 'agent2': True, 'agent3': True},
        },
        'agent2': {
            'env_action': {'alpha': [1, 1, 0]},
            'send': {'agent0': True, 'agent1': True, 'agent3': False},
            'receive': {'agent0': True, 'agent1': True, 'agent3': True},
        },
        'agent3': {
            'env_action': (2, np.array([0, 6]), 1),
            'send': {'agent0': False, 'agent1': False, 'agent2': True},
            'receive': {'agent0': True, 'agent1': True, 'agent2': True},
        }
    }
    env.step(action_0)
    for agent_id in env.agents:
        agent_info = env.get_info(agent_id)
        assert 'send' not in agent_info and 'receive' not in agent_info
    assert env.get_obs('agent0')['env_obs'] == [0, 0, 0, 1]
    assert env.get_obs('agent0')['message_buffer'] == {'agent1': True, 'agent2': True, 'agent3': False}
    assert env.get_obs('agent1')['env_obs'] == 0
    assert env.get_obs('agent1')['message_buffer'] == {'agent0': True, 'agent2': True, 'agent3': False}
    assert env.get_obs('agent2')['env_obs'] == [1, 0]
    assert env.get_obs('agent2')['message_buffer'] == {'agent0': True, 'agent1': False, 'agent3': True}
    assert env.get_obs('agent3')['env_obs'] == {'first': 1, 'second': [3, 1]}
    assert env.get_obs('agent3')['message_buffer'] == {'agent0': True, 'agent1': False, 'agent2': False}

    action_1 = {
        'agent0': {
            'env_action': ({'first': 0, 'second': [3, 3]}, [1, 1, 1]),
            'send': {'agent1': False, 'agent2': False, 'agent3': False},
            'receive': {'agent1': True, 'agent2': False, 'agent3': False},
        },
        'agent1': {
            'env_action': [1, 5, 1],
            'send': {'agent0': True, 'agent2': False, 'agent3': True},
            'receive': {'agent0': False, 'agent2': False, 'agent3': True},
        },
        'agent2': {
            'env_action': {'alpha': [1, 0, 0]},
            'send': {'agent0': True, 'agent1': True, 'agent3': False},
            'receive': {'agent0': True, 'agent1': True, 'agent3': False},
        },
        'agent3': {
            'env_action': (1, np.array([9, 4]), 0),
            'send': {'agent0': False, 'agent1': True, 'agent2': True},
            'receive': {'agent0': True, 'agent1': False, 'agent2': True},
        }
    }
    env.step(action_1)
    assert env.get_obs('agent0')['env_obs'] == [0, 0, 0, 1]
    assert env.get_obs('agent0')['message_buffer'] == {'agent1': True, 'agent2': True, 'agent3': False}
    assert env.get_obs('agent1')['env_obs'] == 0
    assert env.get_obs('agent1')['message_buffer'] == {'agent0': False, 'agent2': True, 'agent3': True}
    assert env.get_obs('agent2')['env_obs'] == [1, 0]
    assert env.get_obs('agent2')['message_buffer'] == {'agent0': False, 'agent1': False, 'agent3': True}
    assert env.get_obs('agent3')['env_obs'] == {'first': 1, 'second': [3, 1]}
    assert env.get_obs('agent3')['message_buffer'] == {'agent0': False, 'agent1': True, 'agent2': False}

    action_2 = {
        'agent0': {
            'env_action': ({'first': 1, 'second': [1, 0]}, [0, 0, 1]),
            'send': {'agent1': True, 'agent2': True, 'agent3': True},
            'receive': {'agent1': False, 'agent2': True, 'agent3': True},
        },
        'agent1': {
            'env_action': [2, 0, 1],
            'send': {'agent0': True, 'agent2': False, 'agent3': False},
            'receive': {'agent0': True, 'agent2': True, 'agent3': True},
        },
        'agent2': {
            'env_action': {'alpha': [0, 0, 0]},
            'send': {'agent0': True, 'agent1': True, 'agent3': False},
            'receive': {'agent0': True, 'agent1': True, 'agent3': False},
        },
        'agent3': {
            'env_action': (0, np.array([7, 7]), 0),
            'send': {'agent0': False, 'agent1': False, 'agent2': True},
            'receive': {'agent0': True, 'agent1': True, 'agent2': False},
        }
    }
    env.step(action_2)
    assert env.get_obs('agent0')['env_obs'] == [0, 0, 1, 0, 0, 1]
    assert env.get_obs('agent0')['message_buffer'] == {'agent1': True, 'agent2': True, 'agent3': False}
    assert env.get_obs('agent1')['env_obs'] == 0
    assert env.get_obs('agent1')['message_buffer'] == {'agent0': True, 'agent2': True, 'agent3': False}
    assert env.get_obs('agent2')['env_obs'] == [1, 0]
    assert env.get_obs('agent2')['message_buffer'] == {'agent0': True, 'agent1': False, 'agent3': True}
    assert env.get_obs('agent3')['env_obs'] == {'first': 0, 'second': [3, 1]}
    assert env.get_obs('agent3')['message_buffer'] == {'agent0': True, 'agent1': False, 'agent2': False}
