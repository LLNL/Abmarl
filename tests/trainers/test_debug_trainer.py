
import os

import pytest
from abmarl.policies.policy import RandomPolicy

from abmarl.trainers import DebugTrainer
from abmarl.managers import AllStepManager, TurnBasedManager
from abmarl.examples import MultiAgentGymSpacesSim


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


def test_debug_trainer_sim(tmpdir):
    with pytest.raises(AssertionError):
        DebugTrainer(
            sim=MultiAgentGymSpacesSim(),
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            output_dir=str(tmpdir)
        )

    trainer = DebugTrainer(
        sim=sim,
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        output_dir=str(tmpdir)
    )

    with pytest.raises(AssertionError):
        trainer.sim = MultiAgentGymSpacesSim()

    trainer.sim = TurnBasedManager(MultiAgentGymSpacesSim())


def test_debug_trainer_policies(tmpdir):
    with pytest.raises(AssertionError):
        DebugTrainer(
            sim=sim,
            policies=[value for value in policies.values()],
            policy_mapping_fn=policy_mapping_fn,
            output_dir=str(tmpdir)
        )
    with pytest.raises(AssertionError):
        DebugTrainer(
            sim=sim,
            policies={int(policy_id[-1]): policy for policy_id, policy in policies.items()},
            policy_mapping_fn=policy_mapping_fn,
            output_dir=str(tmpdir)
        )
    with pytest.raises(AssertionError):
        DebugTrainer(
            sim=sim,
            policies={policy_id: (policy,) for policy_id, policy in policies.items()},
            policy_mapping_fn=policy_mapping_fn,
            output_dir=str(tmpdir)
        )
    DebugTrainer(
        sim=sim,
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        output_dir=str(tmpdir)
    )


def test_debug_trainer_no_policies(tmpdir):
    trainer = DebugTrainer(
        sim=sim, policy_mapping_fn=policy_mapping_fn, output_dir=str(tmpdir)
    )
    for policy_id, policy in trainer.policies.items():
        assert policy_id in sim.agents
        assert isinstance(policy, RandomPolicy)
    assert trainer.policy_mapping_fn('agent0') == 'agent0'
    assert trainer.policy_mapping_fn('agent1') == 'agent1'
    assert trainer.policy_mapping_fn('agent2') == 'agent2'
    assert trainer.policy_mapping_fn('agent3') == 'agent3'


def test_debug_trainer_policy_mapping_fn(tmpdir):
    with pytest.raises(AssertionError):
        DebugTrainer(
            sim=sim,
            policies=policies,
            policy_mapping_fn={'agent0': 'random0'},
            output_dir=str(tmpdir)
        )
    DebugTrainer(
        sim=sim,
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        output_dir=str(tmpdir)
    )


def test_debug_trainer_output_dir(tmpdir):
    debug_trainer = DebugTrainer(
        sim=sim,
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        output_dir=str(tmpdir)
    )
    output = os.path.join(
        os.path.expanduser("~"),
        'abmarl_results',
        tmpdir
    )
    assert debug_trainer.output_dir.startswith(output)


def test_debug_trainer_no_output_dir():
    debug_trainer = DebugTrainer(
        sim=sim,
        policies=policies,
        policy_mapping_fn=policy_mapping_fn
    )
    output = os.path.join(
        os.path.expanduser("~"),
        'abmarl_results',
        'DEBUG_'
    )
    assert debug_trainer.output_dir.startswith(output)


def test_debug_trainer_bad_output_dir():
    with pytest.raises(AssertionError):
        DebugTrainer(
            sim=sim,
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            output_dir=123
        )
