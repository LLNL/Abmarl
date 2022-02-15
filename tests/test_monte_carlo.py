from abmarl.algs.monte_carlo import off_policy
from abmarl.sim.corridor import MultiCorridor as Corridor
from abmarl.managers import AllStepManager
from abmarl.sim.wrappers import RavelDiscreteWrapper
from abmarl.external import GymWrapper
from abmarl.pols.q_table_policy import GreedyPolicy


def test_off_policy():
    sim = AllStepManager(RavelDiscreteWrapper(Corridor(num_agents=1)))
    sim, q_table, policy = off_policy(sim, iteration=100, horizon=10)

    assert isinstance(sim, GymWrapper)
    assert isinstance(sim.sim, AllStepManager)
    assert isinstance(sim.sim.sim, RavelDiscreteWrapper)
    assert isinstance(sim.sim.sim.sim, Corridor)

    assert q_table.shape == (sim.observation_space.n, sim.action_space.n)
    assert isinstance(policy, GreedyPolicy)

    obs = sim.reset()
    for _ in range(10):
        action = policy.compute_action(obs)
        obs, reward, done, info = sim.step(action)
        if done:
            break
    assert done
