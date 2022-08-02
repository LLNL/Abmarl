
import pytest

from abmarl.sim.agent_based_simulation import Agent

from abmarl.trainers import MultiPolicyTrainer
from abmarl.policies.policy import RandomPolicy
from abmarl.managers import AllStepManager, TurnBasedManager
from abmarl.examples import MultiAgentGymSpacesSim


class NoTrainer(MultiPolicyTrainer):
    def train(self, **kwargs):
        return self.generate_episode(horizon=20)


sim = AllStepManager(MultiAgentGymSpacesSim())
policies = {
    'random0': RandomPolicy(
        action_space=sim.agents['agent0'].action_space,
        observation_space=sim.agents['agent0'].observation_space,
    ),
    'random1': RandomPolicy(
        action_space=sim.agents['agent1'].action_space,
        observation_space=sim.agents['agent1'].observation_space,
    ),
    'random2': RandomPolicy(
        action_space=sim.agents['agent2'].action_space,
        observation_space=sim.agents['agent2'].observation_space,
    ),
    'random3': RandomPolicy(
        action_space=sim.agents['agent3'].action_space,
        observation_space=sim.agents['agent3'].observation_space,
    ),
}


def policy_mapping_fn(agent_id):
    return 'random' + agent_id[-1]


def test_trainer_sim():
    with pytest.raises(AssertionError):
        NoTrainer(
            sim=MultiAgentGymSpacesSim(), policies=policies, policy_mapping_fn=policy_mapping_fn
        )

    trainer = NoTrainer(
        sim=sim, policies=policies, policy_mapping_fn=policy_mapping_fn
    )

    with pytest.raises(AssertionError):
        trainer.sim = MultiAgentGymSpacesSim()

    trainer.sim = TurnBasedManager(MultiAgentGymSpacesSim())


def test_trainer_policies():
    with pytest.raises(AssertionError):
        NoTrainer(
            sim=sim,
            policies=[value for value in policies.values()],
            policy_mapping_fn=policy_mapping_fn
        )
    with pytest.raises(AssertionError):
        NoTrainer(
            sim=sim,
            policies={int(policy_id[-1]): policy for policy_id, policy in policies.items()},
            policy_mapping_fn=policy_mapping_fn
        )
    with pytest.raises(AssertionError):
        NoTrainer(
            sim=sim,
            policies={policy_id: (policy,) for policy_id, policy in policies.items()},
            policy_mapping_fn=policy_mapping_fn
        )
    NoTrainer(
        sim=sim, policies=policies, policy_mapping_fn=policy_mapping_fn
    )


def test_trainer_policy_mapping_fn():
    with pytest.raises(AssertionError):
        NoTrainer(
            sim=sim,
            policies=policies,
            policy_mapping_fn={'agent0': 'random0'}
        )
    NoTrainer(
        sim=sim, policies=policies, policy_mapping_fn=policy_mapping_fn
    )


def test_trainer_compute_actions():
    trainer = NoTrainer(
        sim=sim, policies=policies, policy_mapping_fn=policy_mapping_fn
    )
    obs = sim.reset()
    for agent, observation in obs.items():
        policy = trainer.policies[trainer.policy_mapping_fn(agent)]
        assert observation in policy.observation_space
    action = trainer.compute_actions(obs)
    assert len(action) == 4
    for agent in obs:
        assert agent in action
    for agent, act in action.items():
        policy = trainer.policies[trainer.policy_mapping_fn(agent)]
        assert act in policy.action_space


def test_trainer_generate_episode_policy_space_coordination():
    trainer = NoTrainer(
        sim=sim, policies=policies, policy_mapping_fn=policy_mapping_fn
    )
    observations, actions, rewards, dones = trainer.generate_episode(horizon=20)
    for agent_id, observation in observations.items():
        policy = trainer.policies[trainer.policy_mapping_fn(agent_id)]
        assert agent_id in sim.agents
        assert type(observation) is list
        for obs in observation:
            assert obs in policy.observation_space
    for agent_id, action in actions.items():
        policy = trainer.policies[trainer.policy_mapping_fn(agent_id)]
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
        sim=sim, policies=policies, policy_mapping_fn=policy_mapping_fn
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
        sim=sim, policies=policies, policy_mapping_fn=policy_mapping_fn
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
    policies = {
        'random0': RandomPolicy(
            action_space=sim.agents['agent0'].action_space,
            observation_space=sim.agents['agent0'].observation_space,
        ),
        'random1': RandomPolicy(
            action_space=sim.agents['agent0'].action_space,
            observation_space=sim.agents['agent1'].observation_space,
        ),
        'random2': RandomPolicy(
            action_space=sim.agents['agent2'].action_space,
            observation_space=sim.agents['agent2'].observation_space,
        ),
        'random3': RandomPolicy(
            action_space=sim.agents['agent3'].action_space,
            observation_space=sim.agents['agent3'].observation_space,
        ),
    }
    with pytest.raises(AssertionError):
        NoTrainer(sim=sim, policies=policies, policy_mapping_fn=policy_mapping_fn)


def test_policy_observation_space_mismatch():
    policies = {
        'random0': RandomPolicy(
            action_space=sim.agents['agent0'].action_space,
            observation_space=sim.agents['agent0'].observation_space,
        ),
        'random1': RandomPolicy(
            action_space=sim.agents['agent1'].action_space,
            observation_space=sim.agents['agent1'].observation_space,
        ),
        'random2': RandomPolicy(
            action_space=sim.agents['agent2'].action_space,
            observation_space=sim.agents['agent2'].observation_space,
        ),
        'random3': RandomPolicy(
            action_space=sim.agents['agent3'].action_space,
            observation_space=sim.agents['agent0'].observation_space,
        ),
    }
    with pytest.raises(AssertionError):
        NoTrainer(sim=sim, policies=policies, policy_mapping_fn=policy_mapping_fn)
