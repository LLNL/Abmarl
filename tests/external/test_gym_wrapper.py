
from gymnasium import Env as GymEnv
from gymnasium.spaces import Discrete
import pytest

from abmarl.examples.sim.multi_agent_sim import MultiAgentSim
from abmarl.sim.wrappers import RavelDiscreteWrapper
from abmarl.managers import AllStepManager
from abmarl.external import GymWrapper, gym_to_abmarl
from abmarl.tools.gym_utils import Box


class Corridor(GymEnv):
    action_space = Discrete(5)
    observation_space = Box(-1, 4, (1,), int)

    def reset(self, *args, **kwargs):
        return -1, {"info from reset"}

    def step(self, action):
        return action - 1, \
            {0: 0, 1:-1, 2:-2, 3:-3, 4:4}[action], \
            action == 2, \
            action == 3, \
            {"info from step"}


def test_outgoing_gym_wrapper_init():
    sim = MultiAgentSim(1, 4)
    sim_man = AllStepManager(sim)
    agent = sim.agents['agent0']
    gym_wrapped = GymWrapper(sim_man)
    assert gym_wrapped.sim == sim_man
    assert gym_wrapped.agent_id == 'agent0'
    assert gym_wrapped.agent == agent
    assert gym_wrapped.action_space == agent.action_space
    assert gym_wrapped.observation_space == agent.observation_space

    with pytest.raises(AssertionError):
        GymWrapper(sim)
    with pytest.raises(AssertionError):
        GymWrapper(AllStepManager(MultiAgentSim(2, 0)))


def test_outgoing_gym_wrapper_reset_and_step():
    sim = MultiAgentSim(1, 4)
    gym_wrapped = GymWrapper(AllStepManager(sim))

    # Build a new sim so that we don't reference the wrapped one
    sim = AllStepManager(MultiAgentSim(1, 4))
    obs = sim.reset()
    gym_obs, gym_info = gym_wrapped.reset()
    assert obs['agent0'] == gym_obs
    assert gym_info == {}

    action = {'agent0': 0}
    obs, reward, done, info = sim.step(action)
    gym_obs, gym_reward, gym_term, gym_trunc, gym_info = gym_wrapped.step(action['agent0'])
    assert obs['agent0'] == gym_obs
    assert reward['agent0'] == gym_reward
    assert done['agent0'] == gym_term
    assert not gym_trunc
    assert info['agent0'] == gym_info


def test_incoming_gym_wrapper_init():
    sim = Corridor()
    abs = gym_to_abmarl(sim)
    assert abs._gym_env == sim
    assert abs.agents['agent'].observation_space == sim.observation_space
    assert abs.agents['agent'].action_space == sim.action_space
    assert abs.agents['agent'].null_observation == {}
    assert abs.agents['agent'].null_action == {}

    abs = gym_to_abmarl(sim, null_action=0, null_observation=-1)
    assert abs.agents['agent'].null_observation == -1
    assert abs.agents['agent'].null_action == 0

    with pytest.raises(AssertionError):
        gym_to_abmarl(MultiAgentSim(1))


def test_incoming_gym_wrapper_reset_and_step():
    sim = Corridor()
    abmarl_wrapped = AllStepManager(gym_to_abmarl(sim))

    # Build a new sim so that we don't reference the wrapped one
    sim = Corridor()
    obs, _ = sim.reset()
    abmarl_obs = abmarl_wrapped.reset()
    assert obs == abmarl_obs['agent']

    action = 2
    obs, reward, term, trunc, info = sim.step(action)
    abmarl_obs, abmarl_reward, abmarl_done, abmarl_info = abmarl_wrapped.step({'agent': action})
    assert obs == abmarl_obs['agent']
    assert reward == abmarl_reward['agent']
    assert term or trunc == abmarl_done['agent']
    assert info == abmarl_info['agent']

    abmarl_wrapped.reset() # Reset because agent was done
    action = 0
    obs, reward, term, trunc, info = sim.step(action)
    abmarl_obs, abmarl_reward, abmarl_done, abmarl_info = abmarl_wrapped.step({'agent': action})
    assert obs == abmarl_obs['agent']
    assert reward == abmarl_reward['agent']
    assert term or trunc == abmarl_done['agent']
    assert info == abmarl_info['agent']

    action = 3
    obs, reward, term, trunc, info = sim.step(action)
    abmarl_obs, abmarl_reward, abmarl_done, abmarl_info = abmarl_wrapped.step({'agent': action})
    assert obs == abmarl_obs['agent']
    assert reward == abmarl_reward['agent']
    assert term or trunc == abmarl_done['agent']
    assert info == abmarl_info['agent']


def test_incoming_wrap_outgoing():
    sim = Corridor()
    abs = gym_to_abmarl(sim)
    raveled_abs = RavelDiscreteWrapper(abs)
    gym_wrapped = GymWrapper(AllStepManager(raveled_abs))

    assert gym_wrapped.observation_space == Discrete(6)
    assert gym_wrapped.action_space == Discrete(5)
