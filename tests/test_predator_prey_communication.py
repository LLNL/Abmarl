import numpy as np

from abmarl.sim.predator_prey import PredatorPreySimulation, Predator, Prey
from abmarl.sim.wrappers import CommunicationHandshakeWrapper


def test_communication():
    np.random.seed(24)
    agents = [
        Predator(id='predator0', view=1, attack=1),
        Predator(id='predator1', view=8, attack=0),
        Prey(id='prey1', view=4),
        Prey(id='prey2', view=5)
    ]
    sim = PredatorPreySimulation.build(
        {'agents': agents, 'observation_mode': PredatorPreySimulation.ObservationMode.DISTANCE}
    )
    sim = CommunicationHandshakeWrapper(sim)


    sim.reset()
    sim.sim.agents['prey1'].position = np.array([1, 1])
    sim.sim.agents['prey2'].position = np.array([1, 4])
    sim.sim.agents['predator0'].position = np.array([2, 3])
    sim.sim.agents['predator1'].position = np.array([0, 7])

    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['predator1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey2'], np.array([-1, 1, 1])
    )
    assert sim.get_obs('predator0')['message_buffer'] == \
        {'predator1': False, 'prey1': False, 'prey2': False}

    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['predator0'], np.array([2, -4, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey1'], np.array([1, -6, 1])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey2'], np.array([1, -3, 1])
    )
    assert sim.get_obs('predator1')['message_buffer'] == \
        {'predator0': False, 'prey1': False, 'prey2': False}

    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator0'], np.array([1, 2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['prey2'], np.array([0, 3, 1])
    )
    assert sim.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}

    np.testing.assert_array_equal(
        sim.get_obs('prey2')['obs']['predator0'], np.array([1, -1, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey2')['obs']['predator1'], np.array([-1, 3, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey2')['obs']['prey1'], np.array([0, -3, 1])
    )
    assert sim.get_obs('prey2')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey1': False}


    action1 = {
        'predator0': {
            'action': {'move': np.zeros(2), 'attack': 1},
            'send': {'predator1': False, 'prey1': False, 'prey2': False},
            'receive': {'predator1': False, 'prey1': False, 'prey2': True},
        },
        'predator1': {
            'action': {'move': np.zeros(2), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': True},
        },
        'prey1': {
            'action': np.array([-1, 0]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': False, 'predator1': False, 'prey2': False},
        },
        'prey2': {
            'action': np.array([0, 1]),
            'send': {'predator0': False, 'predator1': False, 'prey1': True},
            'receive': {'predator0': True, 'predator1': True, 'prey1': True},
        }
    }
    sim.step(action1)

    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['predator1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator0') == 100
    assert not sim.get_done('predator0')

    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['predator0'], np.array([2, -4, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey1'], np.array([0, -6, 1])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator1')['message_buffer'] == \
        {'predator0': False, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator1') == 0
    assert not sim.get_done('predator1')

    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator0'], np.array([2, 2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': True}
    assert sim.get_reward('prey1') == -1
    assert not sim.get_done('prey1')

    np.testing.assert_array_equal(
        sim.get_obs('prey2')['obs']['predator0'], np.array([1, -1, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey2')['obs']['predator1'], np.array([-1, 3, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey2')['obs']['prey1'], np.array([-1, -3, 1])
    )
    assert sim.get_obs('prey2')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey1': False}
    assert sim.get_reward('prey2') == -100
    assert sim.get_done('prey2')

    assert not sim.get_all_done()


    action2 = {
        'predator0': {
            'action': {'move': np.array([-1, 0]), 'attack': 0},
            'send': {'predator1': False, 'prey1': False, 'prey2': False},
            'receive': {'predator1': False, 'prey1': False, 'prey2': False}
        },
        'predator1': {
            'action': {'move': np.array([1, -1]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'action': np.array([1, -1]),
            'send': {'predator0': False, 'predator1': False, 'prey2': True},
            'receive': {'predator0': False, 'predator1': False, 'prey2': True}
        },
    }
    sim.step(action2)

    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['predator1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator0') == -1
    assert not sim.get_done('predator0')

    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['predator0'], np.array([0, -3, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey1'], np.array([0, -6, 1])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator1')['message_buffer'] == \
        {'predator0': False, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator1') == -1
    assert not sim.get_done('predator1')

    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator0'], np.array([0, 3, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert sim.get_reward('prey1') == -1
    assert not sim.get_done('prey1')


    action3 = {
        'predator0': {
            'action': {'move': np.array([0, -1]), 'attack': 0},
            'send': {'predator1': False, 'prey1': False, 'prey2': False},
            'receive': {'predator1': False, 'prey1': False, 'prey2': False},
        },
        'predator1': {
            'action': {'move': np.array([1, -1]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receieve': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'action': np.array([1, 0]),
            'send': {'predator0': False, 'predator1': False, 'prey2': True},
            'receive': {'predator0': False, 'predator1': False, 'prey2': True}
        }
    }
    sim.step(action3)

    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['predator1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator0') == -1
    assert not sim.get_done('predator0')

    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['predator0'], np.array([-1, -3, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey1'], np.array([0, -5, 1])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator1')['message_buffer'] == \
        {'predator0': False, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator1') == -1
    assert not sim.get_done('predator1')

    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator0'], np.array([-1, 2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert sim.get_reward('prey1') == -1
    assert not sim.get_done('prey1')


    action4 = {
        'predator0': {
            'action': {'move': np.array([0, 0]), 'attack': 1},
            'send': {'predator1': False, 'prey1': False, 'prey2': False},
            'receive': {'predator1': False, 'prey1': True, 'prey2': False},
        },
        'predator1': {
            'action': {'move': np.array([1, -1]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'action': np.array([1, 0]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': True, 'predator1': True, 'prey2': True}
        },
    }
    sim.step(action4)

    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['predator1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator0') == -10
    assert not sim.get_done('predator0')

    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['predator0'], np.array([-2, -2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey1'], np.array([0, -4, 1])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator1')['message_buffer'] == \
        {'predator0': False, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator1') == -1
    assert not sim.get_done('predator1')

    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator0'], np.array([-2, 2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator1'], np.array([0, 4, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert sim.get_reward('prey1') == -1
    assert not sim.get_done('prey1')


    action5 = {
        'predator0': {
            'action': {'move': np.zeros(2), 'attack': 0},
            'send': {'predator1': True, 'prey1': False, 'prey2': False},
            'receive': {'predator1': True, 'prey1': False, 'prey2': False},
        },
        'predator1': {
            'action': {'move': np.array([1, 0]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False},
        },
        'prey1': {
            'action': np.array([0, -1]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': False, 'predator1': False, 'prey2': False},
        },
    }
    sim.step(action5)

    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['predator1'], np.array([3, 2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey1'], np.array([2, -2, 1])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator0') == 0
    assert not sim.get_done('predator0')

    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['predator0'], np.array([-3, -2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey1'], np.array([-1, -4, 1])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator1')['message_buffer'] == \
        {'predator0': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator1') == -1
    assert not sim.get_done('predator1')

    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator0'], np.array([-2, 2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator1'], np.array([1, 4, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert sim.get_reward('prey1') == -10
    assert not sim.get_done('prey1')

    action6 = {
        'predator0': {
            'action': {'move': np.array([1, 0]), 'attack': 0},
            'send': {'predator1': True, 'prey1': False, 'prey2': False},
            'receive': {'predator1': True, 'prey1': False, 'prey2': False}
        },
        'predator1': {
            'action': {'move': np.array([1, 0]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'action': np.array([1, 1]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': False, 'predator1': False, 'prey2': False},
        }
    }
    sim.step(action6)

    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['predator1'], np.array([3, 2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey1'], np.array([2, -1, 1])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator0') == -1
    assert not sim.get_done('predator0')

    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['predator0'], np.array([-3, -2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey1'], np.array([-1, -3, 1])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator1')['message_buffer'] == \
        {'predator0': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator1') == -1
    assert not sim.get_done('predator1')

    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator0'], np.array([-2, 1, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator1'], np.array([1, 3, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert sim.get_reward('prey1') == -1
    assert not sim.get_done('prey1')


    action7 = {
        'predator0': {
            'action': {'move': np.array([1, 0]), 'attack': 0},
            'send': {'predator1': True, 'prey1': False, 'prey2': False},
            'receive': {'predator1': True, 'prey1': False, 'prey2': False}
        },
        'predator1': {
            'action': {'move': np.array([1, 1]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'action': np.array([1, 1]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': False, 'predator1': False, 'prey2': False},
        }
    }
    sim.step(action7)

    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['predator1'], np.array([3, 3, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey1'], np.array([2, 0, 1])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator0') == -1
    assert not sim.get_done('predator0')

    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['predator0'], np.array([-3, -3, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey1'], np.array([-1, -3, 1])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator1')['message_buffer'] == \
        {'predator0': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator1') == -1
    assert not sim.get_done('predator1')

    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator0'], np.array([-2, 0, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator1'], np.array([1, 3, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert sim.get_reward('prey1') == -1
    assert not sim.get_done('prey1')


    action8 = {
        'predator0': {
            'action': {'move': np.array([1, 0]), 'attack': 0},
            'send': {'predator1': True, 'prey1': False, 'prey2': False},
            'receive': {'predator1': True, 'prey1': False, 'prey2': False}
        },
        'predator1': {
            'action': {'move': np.array([-1, -1]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'action': np.array([0, 1]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': False, 'predator1': False, 'prey2': False},
        }
    }
    sim.step(action8)

    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['predator1'], np.array([1, 2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey1'], np.array([1, 1, 1])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator0') == -1
    assert not sim.get_done('predator0')

    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['predator0'], np.array([-1, -2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey1'], np.array([0, -1, 1])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator1')['message_buffer'] == \
        {'predator0': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator1') == -1
    assert not sim.get_done('predator1')

    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator0'], np.array([-1, -1, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator1'], np.array([0, 1, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert sim.get_reward('prey1') == -1
    assert not sim.get_done('prey1')


    action9 = {
        'predator0': {
            'action': {'move': np.array([0, 0]), 'attack': 1},
            'send': {'predator1': True, 'prey1': False, 'prey2': False},
            'receive': {'predator1': True, 'prey1': False, 'prey2': False}
        },
        'predator1': {
            'action': {'move': np.array([0, 0]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'action': np.array([-1, 1]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': False, 'predator1': False, 'prey2': False},
        }
    }
    sim.step(action9)

    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['predator1'], np.array([1, 2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator0')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator0') == 100
    assert not sim.get_done('predator0')

    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['predator0'], np.array([-1, -2, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey1'], np.array([0, 0, 0])
    )
    np.testing.assert_array_equal(
        sim.get_obs('predator1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('predator1')['message_buffer'] == \
        {'predator0': True, 'prey1': False, 'prey2': False}
    assert sim.get_reward('predator1') == 0
    assert not sim.get_done('predator1')

    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator0'], np.array([-1, -1, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['predator1'], np.array([0, 1, 2])
    )
    np.testing.assert_array_equal(
        sim.get_obs('prey1')['obs']['prey2'], np.array([0, 0, 0])
    )
    assert sim.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert sim.get_reward('prey1') == -100
    assert sim.get_done('prey1')

    assert sim.get_all_done()
