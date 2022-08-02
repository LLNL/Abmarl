
from gym.spaces import Discrete, MultiBinary, Box, Dict, Tuple
import pytest

from abmarl.sim.agent_based_simulation import Agent

from abmarl.trainers import SinglePolicyTrainer
from abmarl.policies.policy import RandomPolicy
from abmarl.managers import AllStepManager, TurnBasedManager
from abmarl.examples import MultiAgentSameSpacesSim


class NoTrainer(SinglePolicyTrainer):
    def train(self, **kwargs):
        return self.generate_episode(horizon=20)


sim = AllStepManager(MultiAgentSameSpacesSim())
policy = RandomPolicy(
    action_space=sim.agents['agent0'].action_space,
    observation_space=sim.agents['agent0'].observation_space,
)


def test_trainer_sim():
    with pytest.raises(AssertionError):
        NoTrainer(sim=MultiAgentSameSpacesSim(), policy=policy)

    trainer = NoTrainer(sim=sim, policy=policy)

    with pytest.raises(AssertionError):
        trainer.sim = MultiAgentSameSpacesSim()

    trainer.sim = TurnBasedManager(MultiAgentSameSpacesSim())


def test_trainer_policy():
    with pytest.raises(AssertionError):
        NoTrainer(
            sim=sim,
            policy={'policy': policy}
        )
    trainer = NoTrainer(sim=sim, policy=policy)
    assert trainer.policy == policy
    assert trainer.policies == {'policy': policy}


def test_trainer_policy_mapping_function():
    trainer = NoTrainer(sim=sim, policy=policy)
    assert trainer.policy_mapping_fn('agent0') == 'policy'


def test_trainer_compute_actions():
    trainer = NoTrainer(sim=sim, policy=policy)
    obs = sim.reset()
    for agent, observation in obs.items():
        assert observation in policy.observation_space
    action = trainer.compute_actions(obs)
    assert len(action) == 2
    for agent in obs:
        assert agent in action
    for agent, act in action.items():
        assert act in policy.action_space


def test_trainer_generate_episode_policy_space_coordination():
    trainer = NoTrainer(sim=sim, policy=policy)
    observations, actions, rewards, dones = trainer.generate_episode(horizon=20)
    for agent_id, observation in observations.items():
        assert agent_id in sim.agents
        assert type(observation) is list
        for obs in observation:
            assert obs in policy.observation_space
    for agent_id, action in actions.items():
        assert agent_id in sim.agents
        assert type(action) is list
        for act in action:
            assert act in policy.action_space
    for agent_id, reward in rewards.items():
        assert agent_id in sim.agents
        assert type(reward) is list
        for rew in reward:
            assert type(rew) in [float, int]
    for agent_id, done in dones.items():
        assert agent_id in [*sim.agents, '__all__']
        assert type(done) is list
        for don in done:
            assert type(don) is bool


def test_trainer_generate_episode_check_horizon():
    trainer = NoTrainer(
        sim=sim, policy=policy
    )
    observations, actions, rewards, dones = trainer.generate_episode(horizon=20)
    for obs in observations.values():
        assert len(obs) <= 21
    for action in actions.values():
        assert len(action) <= 20
    for reward in rewards.values():
        assert len(reward) <= 20
    for done in dones.values():
        assert len(done) <= 20


def test_trainer_generate_episode_check_lengths():
    trainer = NoTrainer(
        sim=sim, policy=policy
    )
    observations, actions, rewards, dones = trainer.generate_episode(horizon=20)
    for agent_id, agent in sim.agents.items():
        if not isinstance(agent, Agent): continue
        obs = observations[agent_id]
        action = actions[agent_id]
        reward = rewards[agent_id]
        done = dones[agent_id]
        assert len(obs) == len(action) + 1
        assert len(action) == len(reward) == len(done)


def test_policy_action_space_mismatch():
    policy = RandomPolicy(
        action_space=Tuple((
            Dict({
                'first': Discrete(4),
            }),
            MultiBinary(3)
        )),
        observation_space=MultiBinary(4),
    )
    with pytest.raises(AssertionError):
        NoTrainer(sim=sim, policy=policy)


def test_policy_observation_space_mismatch():
    policy = RandomPolicy(
        action_space=Tuple((
            Dict({
                'first': Discrete(4),
                'second': Box(low=-1, high=3, shape=(2,), dtype=int)
            }),
            MultiBinary(3)
        )),
        observation_space=MultiBinary(5),
    )
    with pytest.raises(AssertionError):
        NoTrainer(sim=sim, policy=policy)
