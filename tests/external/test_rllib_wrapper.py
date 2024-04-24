
from ray.rllib import MultiAgentEnv
from gymnasium.spaces import Discrete, MultiDiscrete, Dict
import numpy as np
import pytest

from abmarl.examples.sim.multi_agent_sim import MultiAgentGymSpacesSim
from abmarl.sim.wrappers import RavelDiscreteWrapper
from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper, multi_agent_to_abmarl
from abmarl.tools.gym_utils import Box


class MultiCorridor(MultiAgentEnv):
    def __init__(self):
        self._agent_ids = {f'agent{i}' for i in range(3)}
        self.observation_space = Dict({
            'agent0': Box(0, 1, (2,), int),
            'agent1': MultiDiscrete([3, 2]),
            'agent2': Box(np.array([-1, 0, 1]), np.array([2, 2, 2]), dtype=int)
        })
        self.action_space = Dict({
            'agent0': Discrete(8),
            'agent1': Box(-3, 3, (2,), int),
            'agent2': MultiDiscrete([2, 2, 2])
        })
        self._action_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True

    def reset(self, *args, **kwargs):
        return {'agent0': [0, 1], 'agent1': [1, 0], 'agent2': [0, 2, 1]}, {"info from reset"}

    def step(self, action):
        obs = {'agent0': [1, 1], 'agent1': [0, 0], 'agent2': [-1, 0, 1]}
        reward = {'agent0': 1, 'agent1': 2, 'agent2': -3}
        term = {'agent0': False, 'agent1': True, 'agent2': False, "__all__": False}
        trunc = {'agent0': False, 'agent1': False, 'agent2': True}
        info = {
            'agent0': 'info from agent0',
            'agent1': 'info from agent1',
            'agent2': 'info from agent2',
        }
        return obs, reward, term, trunc, info


def test_outgoing_multi_agent_wrapper_init():
    sim = MultiAgentGymSpacesSim()
    sim_man = AllStepManager(sim)
    ma_wrapped = MultiAgentWrapper(sim_man)
    assert ma_wrapped.sim == sim_man
    assert ma_wrapped._agent_ids == {'agent0', 'agent1', 'agent2', 'agent3'}
    assert ma_wrapped.observation_space == Dict({
        f'agent{i}': sim.agents[f'agent{i}'].observation_space for i in range(4)
    })
    assert ma_wrapped.action_space == Dict({
        f'agent{i}': sim.agents[f'agent{i}'].action_space for i in range(4)
    })
    assert ma_wrapped._spaces_in_preferred_format

    with pytest.raises(AssertionError):
        MultiAgentWrapper(sim)


def test_outgoing_ma_wrapper_reset_and_step():
    sim = MultiAgentGymSpacesSim()
    ma_wrapped = MultiAgentWrapper(AllStepManager(sim))

    # Build a new sim so that we don't reference the wrapped one
    sim = AllStepManager(MultiAgentGymSpacesSim())
    obs = sim.reset()
    ma_obs, ma_info = ma_wrapped.reset()
    assert obs == ma_obs
    assert ma_info == {}

    action = {f'agent{i}': sim.agents[f'agent{i}'].action_space.sample() for i in range(4)}
    obs, reward, done, info = sim.step(action)
    ma_obs, ma_reward, ma_term, ma_trunc, ma_info = ma_wrapped.step(action)
    assert obs == ma_obs
    assert reward == ma_reward
    assert done == ma_term
    assert ma_trunc == {'__all__': False}
    assert info == ma_info


def test_incoming_ma_wrapper_init():
    sim = MultiCorridor()
    abs = multi_agent_to_abmarl(sim)
    assert abs._env == sim
    for agent in abs.agents.values():
        assert agent.observation_space == sim.observation_space[agent.id]
        assert agent.action_space == sim.action_space[agent.id]
        assert agent.null_observation == {}
        assert agent.null_action == {}

    abs = multi_agent_to_abmarl(
        sim,
        null_action={
            'agent0': ({'first': 0, 'second': [-1, -1]}, [0, 0, 0]),
            'agent1': [0, 0, 0],
            'agent2': {'alpha': [0, 0, 0]},
        },
        null_observation={
            'agent0': [0, 0, 0, 0],
            'agent1': [0],
            'agent2': [0, 0],
        },
    )
    assert abs.agents['agent0'].null_observation == [0, 0, 0, 0]
    assert abs.agents['agent1'].null_observation == [0]
    assert abs.agents['agent2'].null_observation == [0, 0]
    assert abs.agents['agent0'].null_action == ({'first': 0, 'second': [-1, -1]}, [0, 0, 0])
    assert abs.agents['agent1'].null_action == [0, 0, 0]
    assert abs.agents['agent2'].null_action == {'alpha': [0, 0, 0]}


def test_incoming_ma_wrapper_reset_and_step():
    sim = MultiCorridor()
    abmarl_wrapped = AllStepManager(multi_agent_to_abmarl(sim))

    # Build a new sim so that we don't reference the wrapped one
    sim = MultiCorridor()
    obs, _ = sim.reset()
    abmarl_obs = abmarl_wrapped.reset()
    assert obs == abmarl_obs

    action = {f'agent{i}': sim.action_space[f'agent{i}'].sample() for i in range(3)}
    obs, reward, term, trunc, info = sim.step(action)
    abmarl_obs, abmarl_reward, abmarl_done, abmarl_info = abmarl_wrapped.step(action)
    assert obs == abmarl_obs
    assert reward == abmarl_reward
    for agent_id in abmarl_done:
        abmarl_done[agent_id] == term.get(agent_id, False) or trunc.get(agent_id, False)
    assert info == abmarl_info


def test_incoming_wrap_outgoing():
    sim = MultiCorridor()
    abs = multi_agent_to_abmarl(sim)
    raveled_abs = RavelDiscreteWrapper(abs)
    ma_wrapped = MultiAgentWrapper(AllStepManager(raveled_abs))

    assert ma_wrapped.observation_space == Dict({
        'agent0': Discrete(4),
        'agent1': Discrete(6),
        'agent2': Discrete(24)
    })
    assert ma_wrapped.action_space == Dict({
        'agent0': Discrete(8),
        'agent1': Discrete(49),
        'agent2': Discrete(8)
    })
