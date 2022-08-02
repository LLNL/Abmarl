
import numpy as np

from abmarl.algs.monte_carlo import off_policy
from abmarl.examples import MultiCorridor
from abmarl.managers import AllStepManager
from abmarl.sim.wrappers import RavelDiscreteWrapper
from abmarl.external import GymWrapper
from abmarl.policies.q_table_policy import EpsilonSoftPolicy


def test_off_policy():
    np.random.seed(24)
    sim = AllStepManager(RavelDiscreteWrapper(MultiCorridor(num_agents=1)))
    sim, q_table, policy = off_policy(sim, iteration=1000, horizon=10)

    assert isinstance(sim, GymWrapper)
    assert isinstance(sim.sim, AllStepManager)
    assert isinstance(sim.sim.sim, RavelDiscreteWrapper)
    assert isinstance(sim.sim.sim.sim, MultiCorridor)

    assert q_table.shape == (sim.observation_space.n, sim.action_space.n)
    assert isinstance(policy, EpsilonSoftPolicy)

    policy.epsilon = 0
    obs = sim.reset()
    for _ in range(10):
        action = policy.compute_action(obs)
        obs, reward, done, info = sim.step(action)
        if done:
            break
    assert done
