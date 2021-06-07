from abmarl.algs.monte_carlo import exploring_starts, epsilon_soft, off_policy
from abmarl.sim.corridor import MultiCorridor as Corridor
from abmarl.managers import AllStepManager
from abmarl.sim.wrappers import RavelDiscreteWrapper
from abmarl.external import GymWrapper
from abmarl.pols import RandomFirstActionPolicy, EpsilonSoftPolicy, GreedyPolicy


def test_exploring_starts_corridor():
    sim = AllStepManager(RavelDiscreteWrapper(Corridor(num_agents=1)))
    sim, q_table, policy = exploring_starts(sim, iteration=100, horizon=10)

    assert isinstance(sim, GymWrapper)
    assert isinstance(sim.sim, AllStepManager)
    assert isinstance(sim.sim.sim, RavelDiscreteWrapper)
    assert isinstance(sim.sim.sim.sim, Corridor)

    assert q_table.shape == (sim.observation_space.n, sim.action_space.n)
    assert isinstance(policy, RandomFirstActionPolicy)


def test_epsilon_soft():
    sim = AllStepManager(RavelDiscreteWrapper(Corridor(num_agents=1)))
    sim, q_table, policy = epsilon_soft(sim, iteration=1000, horizon=20)

    assert isinstance(sim, GymWrapper)
    assert isinstance(sim.sim, AllStepManager)
    assert isinstance(sim.sim.sim, RavelDiscreteWrapper)
    assert isinstance(sim.sim.sim.sim, Corridor)

    assert q_table.shape == (sim.observation_space.n, sim.action_space.n)
    assert isinstance(policy, EpsilonSoftPolicy)

    obs = sim.reset()
    for _ in range(10):
        action = policy.act(obs)
        obs, reward, done, info = sim.step(action)
        if done:
            break
    assert done


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
        action = policy.act(obs)
        obs, reward, done, info = sim.step(action)
        if done:
            break
    assert done
