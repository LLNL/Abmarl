import pytest

from admiral.managers import AllStepManager
from admiral.external import GymWrapper

from .helpers import MultiAgentEnv


def test_gym_init_multi_agent_error():
    with pytest.raises(AssertionError):
        GymWrapper(MultiAgentEnv(3))


def test_gym_init():
    env = AllStepManager(MultiAgentEnv(1))
    wrapped_env = GymWrapper(env)
    assert wrapped_env.env == env


def test_gym_reset_and_step():
    env = GymWrapper(AllStepManager(MultiAgentEnv(1)))
    obs = env.reset()
    assert obs == 'Obs from agent0'
    obs, reward, done, info = env.step(0)
    assert obs == 'Obs from agent0'
    assert reward == 'Reward from agent0'
    assert done == 'Done from agent0'
    assert info == {'Action from agent0': 0}
