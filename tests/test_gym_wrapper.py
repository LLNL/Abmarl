import pytest

from abmarl.managers import AllStepManager
from abmarl.external import GymWrapper

from .helpers import MultiAgentSim


def test_gym_init_multi_agent_error():
    with pytest.raises(AssertionError):
        GymWrapper(MultiAgentSim(3))


def test_gym_init():
    sim = AllStepManager(MultiAgentSim(1))
    wrapped_sim = GymWrapper(sim)
    assert wrapped_sim.sim == sim


def test_gym_reset_and_step():
    sim = GymWrapper(AllStepManager(MultiAgentSim(1)))
    obs = sim.reset()
    assert obs == 'Obs from agent0'
    obs, reward, done, info = sim.step(0)
    assert obs == 'Obs from agent0'
    assert reward == 'Reward from agent0'
    assert done == 'Done from agent0'
    assert info == {'Action from agent0': 0}
