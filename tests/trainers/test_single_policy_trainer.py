
from gym.spaces import Discrete, MultiBinary, Box, Dict, Tuple
import numpy as np
import pytest

from abmarl.sim.agent_based_simulation import AgentBasedSimulation, Agent, PrincipleAgent

from abmarl.trainers import SinglePolicyTrainer
from abmarl.policies.policy import RandomPolicy
from abmarl.managers import AllStepManager, TurnBasedManager


class SimTest(AgentBasedSimulation):
    def __init__(self):
        self.rewards = [0, 1, 2, 3, 4, 5, 6]
        self.ith_call = -1
        self.dones = [12, 25]
        self.step_count = 0
        self.agents = {
            'agent0': Agent(
                id='agent0',
                observation_space=MultiBinary(4),
                action_space=Tuple((
                    Dict({
                        'first': Discrete(4),
                        'second': Box(low=-1, high=3, shape=(2,), dtype=np.int)
                    }),
                    MultiBinary(3)
                ))
            ),
            'agent1': PrincipleAgent(
                id='agent1',
            ),
            'agent2': Agent(
                id='agent2',
                observation_space=MultiBinary(4),
                action_space=Tuple((
                    Dict({
                        'first': Discrete(4),
                        'second': Box(low=-1, high=3, shape=(2,), dtype=np.int)
                    }),
                    MultiBinary(3)
                ))
            ),
            'agent3': PrincipleAgent(
                id='agent3',
            ),
            'agent4': PrincipleAgent(
                id='agent4'
            )
        }

    def render(self):
        pass

    def reset(self):
        self.action = {
            'agent0': None,
            'agent2': None,
        }

    def step(self, action_dict):
        self.step_count += 1
        for agent_id, action in action_dict.items():
            self.action[agent_id] = action

    def get_obs(self, agent_id, **kwargs):
        if agent_id == 'agent0':
            return [0, 0, 0, 1]
        elif agent_id == 'agent2':
            return [1, 0, 1, 0]

    def get_reward(self, agent_id, **kwargs):
        self.ith_call = (self.ith_call + 1) % 7
        return self.rewards[self.ith_call]

    def get_done(self, agent_id, **kwargs):
        if agent_id == 'agent0':
            return self.step_count >= self.dones[0]
        elif agent_id == "agent2":
            return self.step_count >= self.dones[1]

    def get_all_done(self, **kwargs):
        for agent in self.agents.values():
            if not isinstance(agent, Agent): continue
            if not self.get_done(agent.id):
                return False
        return True

    def get_info(self, agent_id, **kwargs):
        return self.action[agent_id]


class NoTrainer(SinglePolicyTrainer):
    def train(self, **kwargs):
        return self.generate_episode(horizon=20)


sim = AllStepManager(SimTest())
policy = RandomPolicy(
    action_space=sim.agents['agent0'].action_space,
    observation_space=sim.agents['agent0'].observation_space,
)


def test_trainer_sim():
    with pytest.raises(AssertionError):
        NoTrainer(sim=SimTest(), policy=policy)

    trainer = NoTrainer(sim=sim, policy=policy)

    with pytest.raises(AssertionError):
        trainer.sim = SimTest()

    trainer.sim = TurnBasedManager(SimTest())


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
    observations, actions, rewards = trainer.generate_episode(horizon=20)
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


def test_trainer_generate_episode_check_horizon():
    trainer = NoTrainer(
        sim=sim, policy=policy
    )
    observations, actions, rewards = trainer.generate_episode(horizon=20)
    for agent_id, obs in observations.items():
        assert len(obs) <= 21
    for agent_id, action in actions.items():
        assert len(action) <= 20
    for agent_id, reward in rewards.items():
        assert len(reward) <= 20


def test_trainer_generate_episode_check_lengths():
    trainer = NoTrainer(
        sim=sim, policy=policy
    )
    observations, actions, rewards = trainer.generate_episode(horizon=20)
    for agent_id, agent in sim.agents.items():
        if not isinstance(agent, Agent): continue
        obs = observations[agent_id]
        action = actions[agent_id]
        reward = rewards[agent_id]
        assert len(obs) == len(action) + 1
        assert len(action) == len(reward)


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
                'second': Box(low=-1, high=3, shape=(2,), dtype=np.int)
            }),
            MultiBinary(3)
        )),
        observation_space=MultiBinary(5),
    )
    with pytest.raises(AssertionError):
        NoTrainer(sim=sim, policy=policy)
