
from gym.spaces import Discrete, MultiBinary, MultiDiscrete, Box, Dict, Tuple
import pytest

from abmarl.sim.agent_based_simulation import AgentBasedSimulation, Agent, PrincipleAgent

from abmarl.trainers import MultiPolicyTrainer
from abmarl.policies.policy import RandomPolicy
from abmarl.managers import AllStepManager, TurnBasedManager


class SimTest(AgentBasedSimulation):
    def __init__(self):
        self.rewards = [0, 1, 2, 3, 4, 5, 6]
        self.ith_call = -1
        self.dones = [3, 12, 5, 34]
        self.step_count = 0
        self.agents = {
            'agent0': Agent(
                id='agent0',
                observation_space=MultiBinary(4),
                action_space=Tuple((
                    Dict({
                        'first': Discrete(4),
                        'second': Box(low=-1, high=3, shape=(2,), dtype=int)
                    }),
                    MultiBinary(3)
                ))
            ),
            'agent1': Agent(
                id='agent1',
                observation_space=Box(low=0, high=1, shape=(1,), dtype=int),
                action_space=MultiDiscrete([4, 6, 2])
            ),
            'agent2': Agent(
                id='agent2',
                observation_space=MultiDiscrete([2, 2]),
                action_space=Dict({'alpha': MultiBinary(3)})
            ),
            'agent3': Agent(
                id='agent3',
                observation_space=Dict({
                    'first': Discrete(4),
                    'second': Box(low=-1, high=3, shape=(2,), dtype=int)
                }),
                action_space=Tuple((Discrete(3), MultiDiscrete([10, 10]), Discrete(2)))
            ),
            'agent4': PrincipleAgent(
                id='agent4'
            )
        }

    def render(self):
        pass

    def reset(self):
        self.action = {agent.id: None for agent in self.agents.values()}

    def step(self, action_dict):
        self.step_count += 1
        for agent_id, action in action_dict.items():
            self.action[agent_id] = action

    def get_obs(self, agent_id, **kwargs):
        if agent_id == 'agent0':
            return [0, 0, 0, 1]
        elif agent_id == 'agent1':
            return [0]
        elif agent_id == 'agent2':
            return [1, 0]
        elif agent_id == 'agent3':
            return {'first': 1, 'second': [3, 1]}

    def get_reward(self, agent_id, **kwargs):
        self.ith_call = (self.ith_call + 1) % 7
        return self.rewards[self.ith_call]

    def get_done(self, agent_id, **kwargs):
        return self.step_count >= self.dones[int(agent_id[-1])]

    def get_all_done(self, **kwargs):
        for agent in self.agents.values():
            if not isinstance(agent, Agent): continue
            if not self.get_done(agent.id):
                return False
        return True

    def get_info(self, agent_id, **kwargs):
        return self.action[agent_id]


class NoTrainer(MultiPolicyTrainer):
    def train(self, **kwargs):
        return self.generate_episode(horizon=20)


sim = AllStepManager(SimTest())
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
        NoTrainer(sim=SimTest(), policies=policies, policy_mapping_fn=policy_mapping_fn)

    trainer = NoTrainer(
        sim=sim, policies=policies, policy_mapping_fn=policy_mapping_fn
    )

    with pytest.raises(AssertionError):
        trainer.sim = SimTest()

    trainer.sim = TurnBasedManager(SimTest())


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
