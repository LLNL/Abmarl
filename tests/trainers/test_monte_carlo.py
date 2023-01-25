
import numpy as np

from abmarl.examples import MultiCorridor
from abmarl.managers import AllStepManager
from abmarl.sim.wrappers import RavelDiscreteWrapper
from abmarl.policies.q_table_policy import EpsilonSoftPolicy
from abmarl.trainers.monte_carlo import OnPolicyMonteCarloTrainer


def test_epsilon_soft():
    np.random.seed(24)
    sim = AllStepManager(RavelDiscreteWrapper(MultiCorridor(num_agents=1)))
    ref_agent = sim.agents['agent0']
    policy = EpsilonSoftPolicy(
        action_space=ref_agent.action_space,
        observation_space=ref_agent.observation_space
    )
    trainer = OnPolicyMonteCarloTrainer(sim=sim, policy=policy)
    trainer.train(iterations=100, horizon=20)

    policy.epsilon = 0
    obs = sim.reset()
    for _ in range(10):
        actions = trainer.compute_actions(obs)
        obs, reward, done, info = sim.step(actions)
        if done['__all__']:
            break
    assert done['__all__']
