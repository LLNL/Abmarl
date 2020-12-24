
from gym.spaces import Dict, Discrete, Box
import numpy as np
import pytest

from admrial.envs.components.examples.predator_prey import PreyAgent, PredatorAgent, PredatorPreyEnv


from admiral.envs.predator_prey import PredatorPreyEnvDistanceObs, PredatorPreyEnvGridObs
from admiral.envs.predator_prey import Predator, Prey
from admiral.envs.modules import GridResources

def test_direct_distance():
    # Create the agents
    prey =      {f'prey{i}':     PreyAgent(    id=f'prey{i}',     view=2, team=0, move_range=1, max_harvest=0.5) for i in range(2)}
    predators = {f'predator{i}': PredatorAgent(id=f'predator{i}', view=4, team=1, move_range=1, attack_range=1, attack_strength=0.24) for i in range(3)}
    prey['prey0'].starting_position = np.array([1, 1])
    prey['prey1'].starting_position = np.array([3, 2])
    prey['predator0'].starting_position = np.array([4, 4])
    prey['predator1'].starting_position = np.array([3, 3])
    prey['predator2'].starting_position = np.array([7, 9])
    agents = {**prey, **predators}
    
    # Create the environment
    env = PredatorPreyEnv(
        region=10,
        agents=agents,
        number_of_teams=2,
        entropy=0.05
    )
    env.reset()

    # Test the agents' observations
    np.testing.assert_array_equal(env.get_obs('predator0')['position']['predator3'], np.array([-1, -1]))
    np.testing.assert_array_equal(env.get_obs('predator0')['position']['predator4'], np.array([0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['position']['prey0'], np.array([-3, -3]))
    np.testing.assert_array_equal(env.get_obs('predator0')['position']['prey1'], np.array([-1, -2]))

    np.testing.assert_array_equal(env.get_obs('predator1')['position']['predator0'], np.array([1, 1]))
    np.testing.assert_array_equal(env.get_obs('predator1')['position']['predator4'], np.array([0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator1')['position']['prey0'], np.array([-2, -2]))
    np.testing.assert_array_equal(env.get_obs('predator1')['position']['prey1'], np.array([0, -1]))

    np.testing.assert_array_equal(env.get_obs('predator2')['position']['predator0'], np.array([0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator2')['position']['predator1'], np.array([0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator2')['position']['prey0'], np.array([0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator2')['position']['prey1'], np.array([0, 0]))
    
    np.testing.assert_array_equal(env.get_obs('prey0')['position']['predator0'], np.array([0, 0]))
    np.testing.assert_array_equal(env.get_obs('prey0')['position']['predator1'], np.array([2, 2]))
    np.testing.assert_array_equal(env.get_obs('prey0')['position']['predator2'], np.array([0, 0]))
    np.testing.assert_array_equal(env.get_obs('prey0')['position']['prey1'], np.array([2, 1]))
    
    np.testing.assert_array_equal(env.get_obs('prey1')['position']['predator0'], np.array([1, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['position']['predator1'], np.array([0, 1]))
    np.testing.assert_array_equal(env.get_obs('prey1')['position']['predator2'], np.array([0, 0]))
    np.testing.assert_array_equal(env.get_obs('prey1')['position']['prey0'], np.array([-2, -1]))


def test_reset_grid_obs():
    np.random.seed(24)
    agents = [
        Prey(id='prey0', view=2),
        Predator(id='predator1', view=4),
        Prey(id='prey2', view=2),
        Predator(id='predator3', view=4),
        Predator(id='predator4', view=4),
    ]
    env = PredatorPreyEnv.build({'agents': agents})
    env.reset()

    # Explicitly place the agents
    env.agents['predator1'].position = np.array([4,4])
    env.agents['predator3'].position = np.array([3,3])
    env.agents['predator4'].position = np.array([7,9])
    env.agents['prey0'].position = np.array([1,1])
    env.agents['prey2'].position = np.array([3,2])

    assert env.step_count == 0
    np.testing.assert_array_equal(env.get_obs('predator1')['agents'], np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 2., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ]))
    np.testing.assert_array_equal(env.get_obs('predator3')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
    ]))
    np.testing.assert_array_equal(env.get_obs('predator4')['agents'], np.array(
        [[ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    np.testing.assert_array_equal(env.get_obs('prey0')['agents'], np.array(
        [[-1., -1., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  1.,  2.]
    ]))
    np.testing.assert_array_equal(env.get_obs('prey2')['agents'], np.array([
        [0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 2., 0.],
        [0., 0., 0., 0., 2.],
        [0., 0., 0., 0., 0.]
    ]))
    
def test_step_grid_obs():
    np.random.seed(24)
    agents = [
        Predator(id='predator0', view=2, attack=1),
        Prey(id='prey1', view=4),
        Prey(id='prey2', view=5)
    ]
    env = PredatorPreyEnv.build({'agents': agents})
    env.reset()
    env.agents['predator0'].position = np.array([2, 3])
    env.agents['prey1'].position = np.array([0, 7])
    env.agents['prey2'].position = np.array([1, 1])
    np.testing.assert_array_equal(env.get_obs('predator0')['agents'], np.array([
        [0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]
    ]))
    np.testing.assert_array_equal(env.get_obs('prey1')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [ 2.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.]
    ]))
    np.testing.assert_array_equal(env.get_obs('prey2')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1., -1., -1., -1.,  0.,  0.,  0.,  2.,  0.,  0.,  0.],
        [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
    ]))

    action = {
        'predator0': {'move': np.zeros(2), 'attack': 1},
        'prey1': {'move': np.array([0, -1]), 'harvest': 0},
        'prey2': {'move': np.array([-1, 0]), 'harvest': 0},
    }
    env.step(action)
    np.testing.assert_array_equal(env.get_obs('predator0')['agents'], np.array([
        [1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]
    ]))
    assert env.get_reward('predator0') == -10
    assert env.get_done('predator0') == False
    np.testing.assert_array_equal(env.get_obs('prey1')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  2.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.]
    ]))
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False
    np.testing.assert_array_equal(env.get_obs('prey2')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
        [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1., -1., -1., -1.,  0.,  0.,  0.,  2.,  0.,  0.,  0.],
        [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
    ]))
    assert env.get_reward('prey2') == -1
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False

    action = {
        'predator0': {'move': np.array([-1, 0]), 'attack': 0},
        'prey1': {'move': np.array([1, 0]), 'harvest': 0},
        'prey2': {'move': np.array([-1, 0]), 'harvest': 0},
    }
    env.step(action)
    np.testing.assert_array_equal(env.get_obs('predator0')['agents'], np.array([
        [-1., -1., -1., -1., -1.],
        [ 1.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.]]))
    assert env.get_reward('predator0') == -1
    assert env.get_done('predator0') == False
    np.testing.assert_array_equal(env.get_obs('prey1')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
       [ 0.,  2.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.]]))
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False
    np.testing.assert_array_equal(env.get_obs('prey2')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1., -1., -1.,  0.,  0.,  0.,  2.,  0.,  0.,  1.],
       [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]))
    assert env.get_reward('prey2') == -10
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False

    action = {
        'predator0': {'move': np.array([0,0]), 'attack': 0},
        'prey1': {'move': np.array([0, -1]), 'harvest': 0},
        'prey2': {'move': np.array([0, 1]), 'harvest': 0},
    }
    env.step(action)
    np.testing.assert_array_equal(env.get_obs('predator0')['agents'], np.array([
        [-1., -1., -1., -1., -1.],
       [ 0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]]))
    assert env.get_reward('predator0') == 0
    assert env.get_done('predator0') == False
    np.testing.assert_array_equal(env.get_obs('prey1')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  2.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]))
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False
    np.testing.assert_array_equal(env.get_obs('prey2')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1., -1.,  0.,  0.,  0.,  2.,  0.,  1.,  0.,  0.],
       [-1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]))
    assert env.get_reward('prey2') == -1
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False

    action = {
        'predator0': {'move': np.array([0, 1]), 'attack': 0},
        'prey1': {'move': np.array([1, 0]), 'harvest': 0},
        'prey2': {'move': np.array([0, 1]), 'harvest': 0},
    }
    env.step(action)
    np.testing.assert_array_equal(env.get_obs('predator0')['agents'], np.array([
        [-1., -1., -1., -1., -1.],
       [ 0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]]))
    assert env.get_reward('predator0') == -1
    assert env.get_done('predator0') == False
    np.testing.assert_array_equal(env.get_obs('prey1')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]))
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False
    np.testing.assert_array_equal(env.get_obs('prey2')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]))
    assert env.get_reward('prey2') == -1
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False

    action = {
        'predator0': {'move': np.zeros(2), 'attack': 1},
        'prey1': {'move': np.array([1, 0]), 'harvest': 0},
        'prey2': {'move': np.array([1, 0]), 'harvest': 0},
    }
    env.step(action)
    np.testing.assert_array_equal(env.get_obs('predator0')['agents'], np.array([
        [-1., -1., -1., -1., -1.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]]))
    assert env.get_reward('predator0') == 100
    assert env.get_done('predator0') == False
    np.testing.assert_array_equal(env.get_obs('prey1')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  2.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]))
    assert env.get_reward('prey1') == -100
    assert env.get_done('prey1') == True
    np.testing.assert_array_equal(env.get_obs('prey2')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]))
    assert env.get_reward('prey2') == -1
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False

    action = {
        'predator0': {'move': np.zeros(2), 'attack': 1},
        'prey2': {'move': np.array([0, 1]), 'harvest': 0},
    }
    env.step(action)
    np.testing.assert_array_equal(env.get_obs('predator0')['agents'], np.array([
        [-1., -1., -1., -1., -1.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]]))
    assert env.get_reward('predator0') == 100
    assert env.get_done('predator0') == False
    np.testing.assert_array_equal(env.get_obs('prey2')['agents'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]))
    assert env.get_reward('prey2') == -100
    assert env.get_done('prey2') == True
    assert env.get_all_done() == True
    
def test_step_distance_obs():
    np.random.seed(24)
    agents = [
        Predator(id='predator0', view=2, attack=1),
        Prey(id='prey1', view=4),
        Prey(id='prey2', view=5)
    ]
    env = PredatorPreyEnv.build({'agents': agents, 'observation_mode': PredatorPreyEnv.ObservationMode.DISTANCE})
    env.reset()
    env.agents['predator0'].position = np.array([2, 3])
    env.agents['prey1'].position = np.array([0, 7])
    env.agents['prey2'].position = np.array([1, 1])

    np.testing.assert_array_equal(env.get_obs('predator0')['prey1'], np.array([ 0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['prey2'], np.array([-1, -2, 1]))

    np.testing.assert_array_equal(env.get_obs('prey1')['predator0'], np.array([ 2, -4, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['prey2'], np.array([0, 0, 0]))

    np.testing.assert_array_equal(env.get_obs('prey2')['predator0'], np.array([ 1, 2, 2]))
    np.testing.assert_array_equal(env.get_obs('prey2')['prey1'], np.array([0, 0, 0]))


    action = {
        'predator0': {'move': np.zeros(2), 'attack': 1},
        'prey1': np.array([-1, 0]),
        'prey2': np.array([0, 1]),
    }
    env.step(action)

    np.testing.assert_array_equal(env.get_obs('predator0')['prey1'], np.array([ 0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['prey2'], np.array([-1, -1, 1]))
    assert env.get_reward('predator0') == -10
    assert env.get_done('predator0') == False

    np.testing.assert_array_equal(env.get_obs('prey1')['predator0'], np.array([ 2, -4, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['prey2'], np.array([0, 0, 0]))
    assert env.get_reward('prey1') == -10
    assert env.get_done('prey1') == False

    np.testing.assert_array_equal(env.get_obs('prey2')['predator0'], np.array([ 1, 1, 2]))
    np.testing.assert_array_equal(env.get_obs('prey2')['prey1'], np.array([-1, 5, 1]))
    assert env.get_reward('prey2') == -1
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False


    action = {
        'predator0': {'move': np.array([-1, 0]), 'attack': 0},
        'prey1': np.array([0, -1]),
        'prey2': np.array([0, 1]),
    }
    env.step(action)

    np.testing.assert_array_equal(env.get_obs('predator0')['prey1'], np.array([ 0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['prey2'], np.array([0, 0, 1]))
    assert env.get_reward('predator0') == -1
    assert env.get_done('predator0') == False
    
    np.testing.assert_array_equal(env.get_obs('prey1')['predator0'], np.array([ 1, -3, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['prey2'], np.array([1, -3, 1]))
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False

    np.testing.assert_array_equal(env.get_obs('prey2')['predator0'], np.array([ 0, 0, 2]))
    np.testing.assert_array_equal(env.get_obs('prey2')['prey1'], np.array([-1, 3, 1]))
    assert env.get_reward('prey2') == -1
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False


    action = {
        'predator0': {'move': np.array([0,0]), 'attack': 0},
        'prey1': np.array([0, -1]),
        'prey2': np.array([0, 1]),
    }
    env.step(action)

    np.testing.assert_array_equal(env.get_obs('predator0')['prey1'], np.array([-1, 2, 1]))
    np.testing.assert_array_equal(env.get_obs('predator0')['prey2'], np.array([0, 1, 1]))
    assert env.get_reward('predator0') == 0
    assert env.get_done('predator0') == False
    
    np.testing.assert_array_equal(env.get_obs('prey1')['predator0'], np.array([ 1, -2, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['prey2'], np.array([1, -1, 1]))
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False

    np.testing.assert_array_equal(env.get_obs('prey2')['predator0'], np.array([ 0, -1, 2]))
    np.testing.assert_array_equal(env.get_obs('prey2')['prey1'], np.array([-1, 1, 1]))
    assert env.get_reward('prey2') == -1
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False


    action = {
        'predator0': {'move': np.array([0, 1]), 'attack': 0},
        'prey1': np.array([0, -1]),
        'prey2': np.array([-1, 0]),
    }
    env.step(action)

    np.testing.assert_array_equal(env.get_obs('predator0')['prey1'], np.array([-1, 0, 1]))
    np.testing.assert_array_equal(env.get_obs('predator0')['prey2'], np.array([-1, 0, 1]))
    assert env.get_reward('predator0') == -1
    assert env.get_done('predator0') == False

    np.testing.assert_array_equal(env.get_obs('prey1')['predator0'], np.array([ 1, 0, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['prey2'], np.array([0, 0, 1]))
    assert env.get_reward('prey1') == -1
    assert env.get_done('prey1') == False

    np.testing.assert_array_equal(env.get_obs('prey2')['predator0'], np.array([1, 0, 2]))
    np.testing.assert_array_equal(env.get_obs('prey2')['prey1'], np.array([0,0,1]))
    assert env.get_reward('prey2') == -1
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False


    action = {
        'predator0': {'move': np.zeros(2), 'attack': 1},
        'prey1': np.array([0, 1]),
        'prey2': np.array([0, -1]),
    }
    env.step(action)

    np.testing.assert_array_equal(env.get_obs('predator0')['prey1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['prey2'], np.array([-1, -1, 1]))
    assert env.get_reward('predator0') == 100
    assert env.get_done('predator0') == False

    np.testing.assert_array_equal(env.get_obs('prey1')['predator0'], np.array([ 1, 0, 2]))
    np.testing.assert_array_equal(env.get_obs('prey1')['prey2'], np.array([0, -1, 1]))
    assert env.get_reward('prey1') == -100
    assert env.get_done('prey1') == True

    np.testing.assert_array_equal(env.get_obs('prey2')['predator0'], np.array([1, 1, 2]))
    np.testing.assert_array_equal(env.get_obs('prey2')['prey1'], np.array([0,0,0]))
    assert env.get_reward('prey2') == -1
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False


    action = {
        'predator0': {'move': np.zeros(2), 'attack': 1},
        'prey2': np.array([1, 0]),
    }
    env.step(action)

    np.testing.assert_array_equal(env.get_obs('predator0')['prey1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator0')['prey2'], np.array([0, 0, 0]))
    assert env.get_reward('predator0') == 100
    assert env.get_done('predator0') == False

    np.testing.assert_array_equal(env.get_obs('prey2')['predator0'], np.array([1, 1, 2]))
    np.testing.assert_array_equal(env.get_obs('prey2')['prey1'], np.array([0,0,0]))
    assert env.get_reward('prey2') == -100
    assert env.get_done('prey2') == True
    assert env.get_all_done() == True
