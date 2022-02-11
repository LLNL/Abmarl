
from random import Random
from abmarl.sim.corridor import MultiCorridor
from abmarl.managers import AllStepManager
from abmarl.sim.wrappers import RavelDiscreteWrapper
from abmarl.pols import RandomFirstActionPolicy, EpsilonSoftPolicy, GreedyPolicy
from abmarl.trainers.base import SinglePolicyTrainer
from abmarl.trainers.monte_carlo import OnPolicyMonteCarloTrainer


def test_exploring_starts_corridor():
    sim = AllStepManager(RavelDiscreteWrapper(MultiCorridor(num_agents=1)))
    ref_agent = sim.agents['agent0']
    policy = RandomFirstActionPolicy(
        action_space=ref_agent.action_space,
        observation_space=ref_agent.observation_space
    )
    trainer = OnPolicyMonteCarloTrainer(sim=sim, policy=policy)
    trainer.train(iterations=1)
    # sim, q_table, policy = trainer.train(iterations=1)

    # assert isinstance(sim.sim, AllStepManager)
    # assert isinstance(sim.sim.sim, RavelDiscreteWrapper)
    # assert isinstance(sim.sim.sim.sim, MultiCorridor)

    # assert q_table.shape == (sim.observation_space.n, sim.action_space.n)
    # assert isinstance(policy, RandomFirstActionPolicy)
test_exploring_starts_corridor()

# def test_epsilon_soft():
#     sim = AllStepManager(RavelDiscreteWrapper(MultiCorridor(num_agents=1)))
#     sim, q_table, policy = epsilon_soft(sim, iteration=1000, horizon=20)

#     assert isinstance(sim, GymWrapper)
#     assert isinstance(sim.sim, AllStepManager)
#     assert isinstance(sim.sim.sim, RavelDiscreteWrapper)
#     assert isinstance(sim.sim.sim.sim, MultiCorridor)

#     assert q_table.shape == (sim.observation_space.n, sim.action_space.n)
#     assert isinstance(policy, EpsilonSoftPolicy)

#     obs = sim.reset()
#     for _ in range(10):
#         action = policy.act(obs)
#         obs, reward, done, info = sim.step(action)
#         if done:
#             break
#     assert done


# def test_off_policy():
#     sim = AllStepManager(RavelDiscreteWrapper(MultiCorridor(num_agents=1)))
#     sim, q_table, policy = off_policy(sim, iteration=100, horizon=10)

#     assert isinstance(sim, GymWrapper)
#     assert isinstance(sim.sim, AllStepManager)
#     assert isinstance(sim.sim.sim, RavelDiscreteWrapper)
#     assert isinstance(sim.sim.sim.sim, MultiCorridor)

#     assert q_table.shape == (sim.observation_space.n, sim.action_space.n)
#     assert isinstance(policy, GreedyPolicy)

#     obs = sim.reset()
#     for _ in range(10):
#         action = policy.act(obs)
#         obs, reward, done, info = sim.step(action)
#         if done:
#             break
#     assert done
