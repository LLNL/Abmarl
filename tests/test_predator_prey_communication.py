
import numpy as np

from admiral.envs.predator_prey import PredatorPreyEnv, Predator, Prey
from admiral.envs.wrappers import CommunicationWrapper

def test_communication():
    np.random.seed(24)
    agents = [
        Predator(id='predator0', view=1, attack=1),
        Predator(id='predator1', view=8, attack=0),
        Prey(id='prey1', view=4),
        Prey(id='prey2', view=5)
    ]
    env = PredatorPreyEnv.build({'agents': agents, 'observation_mode': PredatorPreyEnv.ObservationMode.DISTANCE})
    env = CommunicationWrapper(env)


    env.reset()
    env.env.agents['prey1'].position = np.array([1, 1])
    env.env.agents['prey2'].position = np.array([1, 4])
    env.env.agents['predator0'].position = np.array([2, 3])
    env.env.agents['predator1'].position = np.array([0, 7])

    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['predator1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey2'], np.array([-1, 1, 1]))
    assert env.get_obs('predator0')['message_buffer'] == \
        {'predator1': False, 'prey1': False, 'prey2': False}

    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['predator0'], np.array([2, -4, 2]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey1'], np.array([1, -6, 1]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey2'], np.array([1, -3, 1]))
    assert env.get_obs('predator1')['message_buffer'] == \
        {'predator0': False, 'prey1': False, 'prey2': False}

    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator0'], np.array([1, 2, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['prey2'], np.array([0, 3, 1]))
    assert env.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}

    np.testing.assert_array_equal(env.get_obs('prey2')['env_obs']['predator0'], np.array([1, -1, 2]))
    np.testing.assert_array_equal(env.get_obs('prey2')['env_obs']['predator1'], np.array([-1, 3, 2]))
    np.testing.assert_array_equal(env.get_obs('prey2')['env_obs']['prey1'], np.array([0, -3, 1]))
    assert env.get_obs('prey2')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey1': False}


    action1 = {
        'predator0': {
            'env_action': {'move': np.zeros(2), 'attack': 1},
            'send': {'predator1': False, 'prey1': False, 'prey2': False},
            'receive': {'predator1': False, 'prey1': False, 'prey2': True},
        },
        'predator1': {
            'env_action': {'move': np.zeros(2), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': True},
        },
        'prey1': {
            'env_action': np.array([-1, 0]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': False, 'predator1': False, 'prey2': False},
        },
        'prey2': {
            'env_action': np.array([0, 1]),
            'send': {'predator0': False, 'predator1': False, 'prey1': True},
            'receive': {'predator0': True, 'predator1': True, 'prey1': True},
        }
    }
    env.step(action1)

    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['predator1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator0') == 100
    assert env.get_done('predator0') == False

    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['predator0'], np.array([2, -4, 2]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey1'], np.array([0, -6, 1]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator1')['message_buffer'] == \
        {'predator0': False, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator1') == 0
    assert env.get_done('predator1') == False

    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator0'], np.array([2, 2, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': True}
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False
        
    np.testing.assert_array_equal(env.get_obs('prey2')['env_obs']['predator0'], np.array([1, -1, 2]))
    np.testing.assert_array_equal(env.get_obs('prey2')['env_obs']['predator1'], np.array([-1, 3, 2]))
    np.testing.assert_array_equal(env.get_obs('prey2')['env_obs']['prey1'], np.array([-1, -3, 1]))
    assert env.get_obs('prey2')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey1': False}
    assert env.get_reward('prey2') == -100
    assert env.get_done('prey2') == True

    assert env.get_all_done() == False


    action2 = {
        'predator0': {
            'env_action': {'move': np.array([-1, 0]), 'attack': 0},
            'send': {'predator1': False, 'prey1': False, 'prey2': False},
            'receive': {'predator1': False, 'prey1': False, 'prey2': False}
        },
        'predator1': {
            'env_action': {'move': np.array([1, -1]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'env_action': np.array([1, -1]),
            'send': {'predator0': False, 'predator1': False, 'prey2': True},
            'receive': {'predator0': False, 'predator1': False, 'prey2': True}
        },
    }
    env.step(action2)

    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['predator1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator0') == -1
    assert env.get_done('predator0') == False

    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['predator0'], np.array([0, -3, 2]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey1'], np.array([0, -6, 1]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator1')['message_buffer'] == \
        {'predator0': False, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator1') == -1
    assert env.get_done('predator1') == False

    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator0'], np.array([0, 3, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False


    action3 = {
        'predator0': {
            'env_action': {'move': np.array([0, -1]), 'attack': 0},
            'send': {'predator1': False, 'prey1': False, 'prey2': False},
            'receive': {'predator1': False, 'prey1': False, 'prey2': False},
        },
        'predator1': {
            'env_action': {'move': np.array([1, -1]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receieve': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'env_action': np.array([1, 0]),
            'send': {'predator0': False, 'predator1': False, 'prey2': True},
            'receive': {'predator0': False, 'predator1': False, 'prey2': True}
        }
    }
    env.step(action3)

    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['predator1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator0') == -1
    assert env.get_done('predator0') == False

    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['predator0'], np.array([-1, -3, 2]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey1'], np.array([0, -5, 1]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator1')['message_buffer'] == \
        {'predator0': False, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator1') == -1
    assert env.get_done('predator1') == False

    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator0'], np.array([-1, 2, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False


    action4 = {
        'predator0': {
            'env_action': {'move': np.array([0, 0]), 'attack': 1},
            'send': {'predator1': False, 'prey1': False, 'prey2': False},
            'receive': {'predator1': False, 'prey1': True, 'prey2': False},
        },
        'predator1': {
            'env_action': {'move': np.array([1, -1]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'env_action': np.array([1, 0]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': True, 'predator1': True, 'prey2': True}
        },
    }
    env.step(action4)

    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['predator1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator0') == -10
    assert env.get_done('predator0') == False

    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['predator0'], np.array([-2, -2, 2]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey1'], np.array([0, -4, 1]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator1')['message_buffer'] == \
        {'predator0': False, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator1') == -1
    assert env.get_done('predator1') == False

    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator0'], np.array([-2, 2, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator1'], np.array([0, 4, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False


    action5 = {
        'predator0': {
            'env_action': {'move': np.zeros(2), 'attack': 0},
            'send': {'predator1': True, 'prey1': False, 'prey2': False},
            'receive': {'predator1': True, 'prey1': False, 'prey2': False},
        },
        'predator1': {
            'env_action': {'move': np.array([1, 0]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False},
        },
        'prey1': {
            'env_action': np.array([0, -1]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': False, 'predator1': False, 'prey2': False},
        },
    }
    env.step(action5)

    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['predator1'], np.array([3, 2, 2]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey1'], np.array([2, -2, 1]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator0') == 0
    assert env.get_done('predator0') == False

    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['predator0'], np.array([-3, -2, 2]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey1'], np.array([-1, -4, 1]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator1')['message_buffer'] == \
        {'predator0': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator1') == -1
    assert env.get_done('predator1') == False

    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator0'], np.array([-2, 2, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator1'], np.array([1, 4, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert env.get_reward('prey1') == -10
    assert env.get_done('prey1') == False

    action6 = {
        'predator0': {
            'env_action': {'move': np.array([1, 0]), 'attack': 0},
            'send': {'predator1': True, 'prey1': False, 'prey2': False},
            'receive': {'predator1': True, 'prey1': False, 'prey2': False}
        },
        'predator1': {
            'env_action': {'move': np.array([1, 0]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'env_action': np.array([1, 1]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': False, 'predator1': False, 'prey2': False},
        }
    }
    env.step(action6)

    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['predator1'], np.array([3, 2, 2]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey1'], np.array([2, -1, 1]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator0') == -1
    assert env.get_done('predator0') == False

    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['predator0'], np.array([-3, -2, 2]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey1'], np.array([-1, -3, 1]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator1')['message_buffer'] == \
        {'predator0': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator1') == -1
    assert env.get_done('predator1') == False

    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator0'], np.array([-2, 1, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator1'], np.array([1, 3, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False


    action7 = {
        'predator0': {
            'env_action': {'move': np.array([1, 0]), 'attack': 0},
            'send': {'predator1': True, 'prey1': False, 'prey2': False},
            'receive': {'predator1': True, 'prey1': False, 'prey2': False}
        },
        'predator1': {
            'env_action': {'move': np.array([1, 1]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'env_action': np.array([1, 1]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': False, 'predator1': False, 'prey2': False},
        }
    }
    env.step(action7)

    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['predator1'], np.array([3, 3, 2]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey1'], np.array([2, 0, 1]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator0') == -1
    assert env.get_done('predator0') == False

    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['predator0'], np.array([-3, -3, 2]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey1'], np.array([-1, -3, 1]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator1')['message_buffer'] == \
        {'predator0': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator1') == -1
    assert env.get_done('predator1') == False

    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator0'], np.array([-2, 0, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator1'], np.array([1, 3, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False


    action8 = {
        'predator0': {
            'env_action': {'move': np.array([1, 0]), 'attack': 0},
            'send': {'predator1': True, 'prey1': False, 'prey2': False},
            'receive': {'predator1': True, 'prey1': False, 'prey2': False}
        },
        'predator1': {
            'env_action': {'move': np.array([-1, -1]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'env_action': np.array([0, 1]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': False, 'predator1': False, 'prey2': False},
        }
    }
    env.step(action8)

    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['predator1'], np.array([1, 2, 2]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey1'], np.array([1, 1, 1]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator0') == -1
    assert env.get_done('predator0') == False

    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['predator0'], np.array([-1, -2, 2]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey1'], np.array([0, -1, 1]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator1')['message_buffer'] == \
        {'predator0': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator1') == -1
    assert env.get_done('predator1') == False

    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator0'], np.array([-1, -1, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator1'], np.array([0, 1, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False


    action9 = {
        'predator0': {
            'env_action': {'move': np.array([0, 0]), 'attack': 1},
            'send': {'predator1': True, 'prey1': False, 'prey2': False},
            'receive': {'predator1': True, 'prey1': False, 'prey2': False}
        },
        'predator1': {
            'env_action': {'move': np.array([0, 0]), 'attack': 0},
            'send': {'predator0': True, 'prey1': False, 'prey2': False},
            'receive': {'predator0': True, 'prey1': True, 'prey2': False}
        },
        'prey1': {
            'env_action': np.array([-1, 1]),
            'send': {'predator0': False, 'predator1': False, 'prey2': False},
            'receive': {'predator0': False, 'predator1': False, 'prey2': False},
        }
    }
    env.step(action9)

    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['predator1'], np.array([1, 2, 2]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator0')['message_buffer'] == \
        {'predator1': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator0') == 100
    assert env.get_done('predator0') == False

    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['predator0'], np.array([-1, -2, 2]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('predator1')['message_buffer'] == \
        {'predator0': True, 'prey1': False, 'prey2': False}
    assert env.get_reward('predator1') == 0
    assert env.get_done('predator1') == False

    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator0'], np.array([-1, -1, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['predator1'], np.array([0, 1, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['env_obs']['prey2'], np.array([0, 0, 0]))
    assert env.get_obs('prey1')['message_buffer'] == \
        {'predator0': False, 'predator1': False, 'prey2': False}
    assert env.get_reward('prey1') == -100
    assert env.get_done('prey1') == True

    assert env.get_all_done() == True
