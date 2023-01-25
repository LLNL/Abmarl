
from open_spiel.python.algorithms import random_agent
from open_spiel.python.rl_environment import TimeStep, StepType
import pytest

from abmarl.examples import MultiCorridor
from abmarl.sim.wrappers import RavelDiscreteWrapper
from abmarl.managers import AllStepManager, TurnBasedManager
from abmarl.external import OpenSpielWrapper

abs = RavelDiscreteWrapper(
    MultiCorridor()
)
agents = abs.agents

trainers = {
    agent.id: random_agent.RandomAgent(
        player_id=agent.id,
        num_actions=agent.action_space.n
    )
    for agent in agents.values()
}


def test_wrapper():
    sim = OpenSpielWrapper(
        AllStepManager(abs)
    )
    assert sim._learning_agents == abs.agents
    assert sim.sim.sim == abs
    assert sim.discounts == {
        'agent0': 1.0,
        'agent1': 1.0,
        'agent2': 1.0,
        'agent3': 1.0,
        'agent4': 1.0,
    }
    assert sim._should_reset
    assert sim.num_players == 5
    assert not sim.is_turn_based

    assert sim.observation_spec() == {
        agent.id: {
            'info_state': (agent.observation_space.n,),
            'legal_actions': (agent.action_space.n,),
            'current_player': (),
        } for agent in agents.values()
    }
    assert sim.action_spec() == {
        agent.id: {
            'num_actions': agent.action_space.n,
            'min': 0,
            'max': agent.action_space.n - 1,
            'dtype': int
        } for agent in agents.values()
    }


def test_sim():
    with pytest.raises(AssertionError):
        OpenSpielWrapper(TurnBasedManager(MultiCorridor()))

    with pytest.raises(AssertionError):
        OpenSpielWrapper(abs)


def test_discounts():
    with pytest.raises(AssertionError):
        OpenSpielWrapper(
            AllStepManager(MultiCorridor()),
            discounts='str'
        )
    with pytest.raises(AssertionError):
        OpenSpielWrapper(
            AllStepManager(MultiCorridor()),
            discounts={
                'agent0': 0.1,
                'agent1': 0.5,
                'agent2': 'str',
                'agent3': 0.6,
                'agent4': 1.0,
            }
        )
    with pytest.raises(AssertionError):
        OpenSpielWrapper(
            AllStepManager(MultiCorridor()),
            discounts={
                'agent0': 0.1,
                'agent1': 0.5,
                'agent2': 1.0,
                'agent3': 0.6,
                'agent4': 1.0,
                'agent5': 0.9,
            }
        )
    with pytest.raises(AssertionError):
        OpenSpielWrapper(
            AllStepManager(MultiCorridor()),
            discounts={
                'agent0': 0.1,
                'agent1': 0.5,
                'agent2': 1.0,
                'agent3': 0.6,
            }
        )


def test_wrapper_reset():
    sim = OpenSpielWrapper(
        AllStepManager(abs)
    )
    time_step = sim.reset()
    assert not sim._should_reset
    assert sim.current_player == 'agent0'
    assert isinstance(time_step, TimeStep)
    assert time_step.observations.keys() == set({'info_state', 'legal_actions', 'current_player'})
    assert time_step.observations['info_state'] == {
        'agent0': sim.sim.sim.get_obs('agent0'),
        'agent1': sim.sim.sim.get_obs('agent1'),
        'agent2': sim.sim.sim.get_obs('agent2'),
        'agent3': sim.sim.sim.get_obs('agent3'),
        'agent4': sim.sim.sim.get_obs('agent4'),
    }
    assert time_step.observations['legal_actions'] == {
        'agent0': [0, 1, 2],
        'agent1': [0, 1, 2],
        'agent2': [0, 1, 2],
        'agent3': [0, 1, 2],
        'agent4': [0, 1, 2],
    }
    assert time_step.observations['current_player'] == sim.current_player
    assert time_step.rewards is None
    assert time_step.discounts is None
    assert time_step.step_type == StepType.FIRST


def test_wrapper_step():
    sim = OpenSpielWrapper(
        AllStepManager(abs)
    )
    sim.reset()
    action_list = [0, 1, 2, 2, 1]
    time_step = sim.step(action_list)
    assert isinstance(time_step, TimeStep)
    assert time_step.observations.keys() == set({'info_state', 'legal_actions', 'current_player'})
    assert time_step.observations['info_state'] == {
        'agent0': sim.sim.sim.get_obs('agent0'),
        'agent1': sim.sim.sim.get_obs('agent1'),
        'agent2': sim.sim.sim.get_obs('agent2'),
        'agent3': sim.sim.sim.get_obs('agent3'),
        'agent4': sim.sim.sim.get_obs('agent4'),
    }
    assert time_step.observations['legal_actions'] == {
        'agent0': [0, 1, 2],
        'agent1': [0, 1, 2],
        'agent2': [0, 1, 2],
        'agent3': [0, 1, 2],
        'agent4': [0, 1, 2],
    }
    assert time_step.observations['current_player'] == sim.current_player
    assert time_step.rewards.keys() == set({'agent0', 'agent1', 'agent2', 'agent3', 'agent4'})
    assert time_step.discounts == sim._discounts
    assert time_step.step_type == StepType.MID


def test_step_mismatched_number():
    sim = OpenSpielWrapper(
        AllStepManager(abs)
    )
    sim.reset()
    with pytest.raises(AssertionError):
        action_list = [0, 1, 2, 2]
        sim.step(action_list)


def test_take_fake_step():
    sim = OpenSpielWrapper(
        AllStepManager(abs)
    )
    sim.reset()
    sim.sim.done_agents = set({'agent0', 'agent1', 'agent2', 'agent3', 'agent4'})
    time_step = sim.step([0, 1, 2, 2, 1])
    assert isinstance(time_step, TimeStep)
    assert time_step.observations.keys() == set({'info_state', 'legal_actions', 'current_player'})
    assert time_step.observations['info_state'] == {
        'agent0': sim.sim.sim.get_obs('agent0'),
        'agent1': sim.sim.sim.get_obs('agent1'),
        'agent2': sim.sim.sim.get_obs('agent2'),
        'agent3': sim.sim.sim.get_obs('agent3'),
        'agent4': sim.sim.sim.get_obs('agent4'),
    }
    assert time_step.observations['legal_actions'] == {
        'agent0': [0, 1, 2],
        'agent1': [0, 1, 2],
        'agent2': [0, 1, 2],
        'agent3': [0, 1, 2],
        'agent4': [0, 1, 2],
    }
    assert time_step.observations['current_player'] == sim.current_player
    assert time_step.rewards.keys() == set({'agent0', 'agent1', 'agent2', 'agent3', 'agent4'})
    assert time_step.discounts == sim._discounts
    assert time_step.step_type == StepType.MID


def test_should_reset():
    sim = OpenSpielWrapper(
        AllStepManager(abs)
    )
    assert sim._should_reset
    time_step = sim.reset()
    assert not sim._should_reset
    assert time_step.step_type == StepType.FIRST

    time_step = sim.step([0, 1, 2, 2, 1])
    assert time_step.step_type == StepType.MID

    sim._should_reset = True
    time_step = sim.step([0, 1, 2, 2, 1])
    assert time_step.step_type == StepType.FIRST


def test_rl_main_loop_all_step():
    sim = OpenSpielWrapper(
        AllStepManager(abs)
    )
    assert not sim.is_turn_based
    for _ in range(5):
        time_step = sim.reset()
        for _ in range(20):
            agents_output = [trainer.step(time_step) for trainer in trainers.values()]
            action_list = [agent_output.action for agent_output in agents_output]
            assert len(action_list) == 5
            time_step = sim.step(action_list)
            if time_step.last():
                for trainer in trainers.values():
                    trainer.step(time_step)
                    break
        for trainer in trainers.values():
            trainer.step(time_step)


def test_rl_main_loop_turn_based():
    sim = OpenSpielWrapper(
        TurnBasedManager(abs)
    )
    assert sim.is_turn_based
    for _ in range(5):
        time_step = sim.reset()
        for _ in range(20):
            player_id = time_step.observations["current_player"]
            agent_output = trainers[player_id].step(time_step)
            action_list = [agent_output.action]
            assert len(action_list) == 1
            time_step = sim.step(action_list)
            if time_step.last():
                for trainer in trainers.values():
                    trainer.step(time_step)
                    break
        for trainer in trainers.values():
            trainer.step(time_step)
