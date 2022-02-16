
from abmarl.sim.corridor import MultiCorridor
from abmarl.managers import AllStepManager
from abmarl.sim.wrappers import RavelDiscreteWrapper
from abmarl.policies.q_table_policy import EpsilonSoftPolicy
from abmarl.trainers.monte_carlo import OnPolicyMonteCarloTrainer


def test_epsilon_soft():
    sim = AllStepManager(RavelDiscreteWrapper(MultiCorridor(num_agents=1)))
    ref_agent = sim.agents['agent0']
    policy = EpsilonSoftPolicy(
        action_space=ref_agent.action_space,
        observation_space=ref_agent.observation_space
    )
    trainer = OnPolicyMonteCarloTrainer(sim=sim, policy=policy)
    trainer.train(iterations=2000, horizon=20)

    obs = sim.reset()
    for _ in range(20):
        actions = trainer.compute_actions(obs)
        obs, reward, done, info = sim.step(actions)
        if done['__all__']:
            break
    assert done['__all__']
