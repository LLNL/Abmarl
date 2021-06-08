import numpy as np
import pytest

from abmarl.sim.predator_prey import PredatorPreySimulation, Predator, Prey
from abmarl.managers import AllStepManager


def test_turn_based_predator_prey_distance():
    np.random.seed(24)
    predators = [Predator(id=f'predator{i}', attack=1) for i in range(2)]
    prey = [Prey(id=f'prey{i}') for i in range(7)]
    agents = predators + prey
    sim_config = {
        'region': 6,
        'observation_mode': PredatorPreySimulation.ObservationMode.DISTANCE,
        'agents': agents,
    }
    sim = PredatorPreySimulation.build(sim_config)
    sim = AllStepManager(sim)

    # Little hackish here because I have to explicitly set their values
    obs = sim.reset()
    sim.agents['predator0'].position = np.array([2, 3])
    sim.agents['predator1'].position = np.array([0, 1])
    sim.agents['prey0'].position = np.array([1, 1])
    sim.agents['prey1'].position = np.array([4, 3])
    sim.agents['prey2'].position = np.array([4, 3])
    sim.agents['prey3'].position = np.array([2, 3])
    sim.agents['prey4'].position = np.array([3, 3])
    sim.agents['prey5'].position = np.array([3, 1])
    sim.agents['prey6'].position = np.array([2, 1])
    obs = {agent_id: sim.sim.get_obs(agent_id) for agent_id in sim.agents}

    np.testing.assert_array_equal(obs['predator0']['predator1'], np.array([-2, -2,  2]))
    np.testing.assert_array_equal(obs['predator0']['prey0'], np.array([-1, -2,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey1'], np.array([2, 0,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey2'], np.array([2, 0,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey3'], np.array([0, 0,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey4'], np.array([1, 0,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey5'], np.array([1, -2,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey6'], np.array([0, -2,  1]))

    np.testing.assert_array_equal(obs['predator1']['predator0'], np.array([2, 2,  2]))
    np.testing.assert_array_equal(obs['predator1']['prey0'], np.array([1, 0,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey1'], np.array([4, 2,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey2'], np.array([4, 2,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey3'], np.array([2, 2,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey4'], np.array([3, 2,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey5'], np.array([3, 0,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey6'], np.array([2, 0,  1]))

    np.testing.assert_array_equal(obs['prey0']['predator0'], np.array([1, 2,  2]))
    np.testing.assert_array_equal(obs['prey0']['predator1'], np.array([-1, 0,  2]))
    np.testing.assert_array_equal(obs['prey0']['prey1'], np.array([3, 2,  1]))
    np.testing.assert_array_equal(obs['prey0']['prey2'], np.array([3, 2,  1]))
    np.testing.assert_array_equal(obs['prey0']['prey3'], np.array([1, 2, 1]))
    np.testing.assert_array_equal(obs['prey0']['prey4'], np.array([2, 2,  1]))
    np.testing.assert_array_equal(obs['prey0']['prey5'], np.array([2, 0,  1]))
    np.testing.assert_array_equal(obs['prey0']['prey6'], np.array([1, 0,  1]))

    np.testing.assert_array_equal(obs['prey1']['predator0'], np.array([-2, 0,  2]))
    np.testing.assert_array_equal(obs['prey1']['predator1'], np.array([-4, -2,  2]))
    np.testing.assert_array_equal(obs['prey1']['prey0'], np.array([-3, -2,  1]))
    np.testing.assert_array_equal(obs['prey1']['prey2'], np.array([0, 0,  1]))
    np.testing.assert_array_equal(obs['prey1']['prey3'], np.array([-2, 0, 1]))
    np.testing.assert_array_equal(obs['prey1']['prey4'], np.array([-1, 0,  1]))
    np.testing.assert_array_equal(obs['prey1']['prey5'], np.array([-1, -2,  1]))
    np.testing.assert_array_equal(obs['prey1']['prey6'], np.array([-2, -2,  1]))

    np.testing.assert_array_equal(obs['prey2']['predator0'], np.array([-2, 0,  2]))
    np.testing.assert_array_equal(obs['prey2']['predator1'], np.array([-4, -2,  2]))
    np.testing.assert_array_equal(obs['prey2']['prey0'], np.array([-3, -2, 1]))
    np.testing.assert_array_equal(obs['prey2']['prey1'], np.array([0, 0,  1]))
    np.testing.assert_array_equal(obs['prey2']['prey3'], np.array([-2, 0, 1]))
    np.testing.assert_array_equal(obs['prey2']['prey4'], np.array([-1, 0,  1]))
    np.testing.assert_array_equal(obs['prey2']['prey5'], np.array([-1, -2,  1]))
    np.testing.assert_array_equal(obs['prey2']['prey6'], np.array([-2, -2,  1]))

    np.testing.assert_array_equal(obs['prey3']['predator0'], np.array([0, 0,  2]))
    np.testing.assert_array_equal(obs['prey3']['predator1'], np.array([-2, -2,  2]))
    np.testing.assert_array_equal(obs['prey3']['prey0'], np.array([-1, -2, 1]))
    np.testing.assert_array_equal(obs['prey3']['prey1'], np.array([2, 0,  1]))
    np.testing.assert_array_equal(obs['prey3']['prey2'], np.array([2, 0, 1]))
    np.testing.assert_array_equal(obs['prey3']['prey4'], np.array([1, 0,  1]))
    np.testing.assert_array_equal(obs['prey3']['prey5'], np.array([1, -2,  1]))
    np.testing.assert_array_equal(obs['prey3']['prey6'], np.array([0, -2,  1]))

    np.testing.assert_array_equal(obs['prey4']['predator0'], np.array([-1, 0,  2]))
    np.testing.assert_array_equal(obs['prey4']['predator1'], np.array([-3, -2,  2]))
    np.testing.assert_array_equal(obs['prey4']['prey0'], np.array([-2, -2, 1]))
    np.testing.assert_array_equal(obs['prey4']['prey1'], np.array([1, 0,  1]))
    np.testing.assert_array_equal(obs['prey4']['prey2'], np.array([1, 0,  1]))
    np.testing.assert_array_equal(obs['prey4']['prey3'], np.array([-1, 0, 1]))
    np.testing.assert_array_equal(obs['prey4']['prey5'], np.array([0, -2,  1]))
    np.testing.assert_array_equal(obs['prey4']['prey6'], np.array([-1, -2,  1]))

    np.testing.assert_array_equal(obs['prey5']['predator0'], np.array([-1, 2,  2]))
    np.testing.assert_array_equal(obs['prey5']['predator1'], np.array([-3, 0,  2]))
    np.testing.assert_array_equal(obs['prey5']['prey0'], np.array([-2, 0, 1]))
    np.testing.assert_array_equal(obs['prey5']['prey1'], np.array([1, 2,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey2'], np.array([1, 2,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey3'], np.array([-1, 2, 1]))
    np.testing.assert_array_equal(obs['prey5']['prey4'], np.array([0, 2,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey6'], np.array([-1, 0,  1]))

    np.testing.assert_array_equal(obs['prey6']['predator0'], np.array([0, 2,  2]))
    np.testing.assert_array_equal(obs['prey6']['predator1'], np.array([-2, 0,  2]))
    np.testing.assert_array_equal(obs['prey6']['prey0'], np.array([-1, 0, 1]))
    np.testing.assert_array_equal(obs['prey6']['prey1'], np.array([2, 2,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey2'], np.array([2, 2,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey3'], np.array([0, 2, 1]))
    np.testing.assert_array_equal(obs['prey6']['prey4'], np.array([1, 2,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey5'], np.array([1, 0, 1]))


    obs, reward, done, info = sim.step({
        'predator0': {'attack': 1, 'move': np.array([0, 0])},
        'predator1': {'attack': 1, 'move': np.array([0, 0])},
        'prey0': np.array([-1, 1]),
        'prey1': np.array([0, -1]),
        'prey2': np.array([1, 1]),
        'prey3': np.array([1, -1]),
        'prey4': np.array([-1, 1]),
        'prey5': np.array([1, 1]),
        'prey6': np.array([0, 0]),
    })

    np.testing.assert_array_equal(obs['predator0']['predator1'], np.array([-2, -2,  2]))
    np.testing.assert_array_equal(obs['predator0']['prey0'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['predator0']['prey1'], np.array([2, -1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey2'], np.array([3, 1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey4'], np.array([0, 1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey5'], np.array([2, -1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey6'], np.array([0, -2,  1]))

    np.testing.assert_array_equal(obs['predator1']['predator0'], np.array([2, 2,  2]))
    np.testing.assert_array_equal(obs['predator1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey1'], np.array([4, 1,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey2'], np.array([5, 3,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey4'], np.array([2, 3, 1]))
    np.testing.assert_array_equal(obs['predator1']['prey5'], np.array([4, 1,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey6'], np.array([2, 0,  1]))

    np.testing.assert_array_equal(obs['prey0']['predator0'], np.array([1, 2,  2]))
    np.testing.assert_array_equal(obs['prey0']['predator1'], np.array([-1, 0,  2]))
    np.testing.assert_array_equal(obs['prey0']['prey1'], np.array([3, 1,  1]))
    np.testing.assert_array_equal(obs['prey0']['prey2'], np.array([4, 3,  1]))
    np.testing.assert_array_equal(obs['prey0']['prey3'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['prey0']['prey4'], np.array([1, 3,  1]))
    np.testing.assert_array_equal(obs['prey0']['prey5'], np.array([3, 1,  1]))
    np.testing.assert_array_equal(obs['prey0']['prey6'], np.array([1, 0,  1]))

    np.testing.assert_array_equal(obs['prey1']['predator0'], np.array([-2, 1,  2]))
    np.testing.assert_array_equal(obs['prey1']['predator1'], np.array([-4, -1,  2]))
    np.testing.assert_array_equal(obs['prey1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey2'], np.array([1, 2,  1]))
    np.testing.assert_array_equal(obs['prey1']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey4'], np.array([-2, 2, 1]))
    np.testing.assert_array_equal(obs['prey1']['prey5'], np.array([0, 0,  1]))
    np.testing.assert_array_equal(obs['prey1']['prey6'], np.array([-2, -1,  1]))

    np.testing.assert_array_equal(obs['prey2']['predator0'], np.array([-3, -1,  2]))
    np.testing.assert_array_equal(obs['prey2']['predator1'], np.array([-5, -3,  2]))
    np.testing.assert_array_equal(obs['prey2']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey1'], np.array([-1, -2,  1]))
    np.testing.assert_array_equal(obs['prey2']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey4'], np.array([-3, 0, 1]))
    np.testing.assert_array_equal(obs['prey2']['prey5'], np.array([-1, -2,  1]))
    np.testing.assert_array_equal(obs['prey2']['prey6'], np.array([-3, -3,  1]))

    np.testing.assert_array_equal(obs['prey3']['predator0'], np.array([0, 0, 2]))
    np.testing.assert_array_equal(obs['prey3']['predator1'], np.array([-2, -2,  2]))
    np.testing.assert_array_equal(obs['prey3']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey3']['prey1'], np.array([2, -1,  1]))
    np.testing.assert_array_equal(obs['prey3']['prey2'], np.array([3, 1, 1]))
    np.testing.assert_array_equal(obs['prey3']['prey4'], np.array([0, 1, 1]))
    np.testing.assert_array_equal(obs['prey3']['prey5'], np.array([2, -1,  1]))
    np.testing.assert_array_equal(obs['prey3']['prey6'], np.array([0, -2,  1]))

    np.testing.assert_array_equal(obs['prey4']['predator0'], np.array([0, -1,  2]))
    np.testing.assert_array_equal(obs['prey4']['predator1'], np.array([-2, -3,  2]))
    np.testing.assert_array_equal(obs['prey4']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey4']['prey1'], np.array([2, -2,  1]))
    np.testing.assert_array_equal(obs['prey4']['prey2'], np.array([3, 0,  1]))
    np.testing.assert_array_equal(obs['prey4']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey4']['prey5'], np.array([2, -2,  1]))
    np.testing.assert_array_equal(obs['prey4']['prey6'], np.array([0, -3,  1]))

    np.testing.assert_array_equal(obs['prey5']['predator0'], np.array([-2, 1,  2]))
    np.testing.assert_array_equal(obs['prey5']['predator1'], np.array([-4, -1,  2]))
    np.testing.assert_array_equal(obs['prey5']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey5']['prey1'], np.array([0, 0,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey2'], np.array([1, 2,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey5']['prey4'], np.array([-2, 2, 1]))
    np.testing.assert_array_equal(obs['prey5']['prey6'], np.array([-2, -1,  1]))

    np.testing.assert_array_equal(obs['prey6']['predator0'], np.array([0, 2,  2]))
    np.testing.assert_array_equal(obs['prey6']['predator1'], np.array([-2, 0,  2]))
    np.testing.assert_array_equal(obs['prey6']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey1'], np.array([2, 1,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey2'], np.array([3, 3,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey4'], np.array([0, 3, 1]))
    np.testing.assert_array_equal(obs['prey6']['prey5'], np.array([2, 1,  1]))

    assert reward == {
        'predator0': 36,
        'predator1': 36,
        'prey0': -36,
        'prey1': -1,
        'prey2': -1,
        'prey3': -36,
        'prey4': -1,
        'prey5': -1,
        'prey6': 0,
    }

    assert done == {
        'predator0': False,
        'predator1': False,
        'prey0': True,
        'prey1': False,
        'prey2': False,
        'prey3': True,
        'prey4': False,
        'prey5': False,
        'prey6': False,
        '__all__': False}

    with pytest.raises(AssertionError):
        obs, reward, done, info = sim.step({
            'predator0': {'attack': 1, 'move': np.array([0, 0])},
            'predator1': {'attack': 1, 'move': np.array([0, 0])},
            'prey0': np.array([-1, 1]),
            'prey1': np.array([0, -1]),
            'prey2': np.array([1, 1]),
            'prey3': np.array([1, -1]),
            'prey4': np.array([-1, 1]),
            'prey5': np.array([1, 1]),
            'prey6': np.array([0, 0]),
        })

    obs, reward, done, info = sim.step({
        'predator0': {'attack': 1, 'move': np.array([0, 0])},
        'predator1': {'attack': 0, 'move': np.array([1, 0])},
        'prey1': np.array([-1, -1]),
        'prey2': np.array([-1, 0]),
        'prey4': np.array([-1, 0]),
        'prey5': np.array([-1, 0]),
        'prey6': np.array([0, -1]),
    })

    np.testing.assert_array_equal(obs['predator0']['predator1'], np.array([-1, -2,  2]))
    np.testing.assert_array_equal(obs['predator0']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey1'], np.array([1, -2,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey2'], np.array([2, 1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey5'], np.array([1, -1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey6'], np.array([0, -3,  1]))

    np.testing.assert_array_equal(obs['predator1']['predator0'], np.array([1, 2,  2]))
    np.testing.assert_array_equal(obs['predator1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey1'], np.array([2, 0,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey2'], np.array([3, 3,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey5'], np.array([2, 1, 1]))
    np.testing.assert_array_equal(obs['predator1']['prey6'], np.array([1, -1,  1]))

    np.testing.assert_array_equal(obs['prey1']['predator0'], np.array([-1, 2,  2]))
    np.testing.assert_array_equal(obs['prey1']['predator1'], np.array([-2, 0,  2]))
    np.testing.assert_array_equal(obs['prey1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey2'], np.array([1, 3,  1]))
    np.testing.assert_array_equal(obs['prey1']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey5'], np.array([0, 1, 1]))
    np.testing.assert_array_equal(obs['prey1']['prey6'], np.array([-1, -1, 1]))

    np.testing.assert_array_equal(obs['prey2']['predator0'], np.array([-2, -1,  2]))
    np.testing.assert_array_equal(obs['prey2']['predator1'], np.array([-3, -3,  2]))
    np.testing.assert_array_equal(obs['prey2']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey1'], np.array([-1, -3,  1]))
    np.testing.assert_array_equal(obs['prey2']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey5'], np.array([-1, -2, 1]))
    np.testing.assert_array_equal(obs['prey2']['prey6'], np.array([-2, -4, 1]))

    np.testing.assert_array_equal(obs['prey4']['predator0'], np.array([0, -1,  2]))
    np.testing.assert_array_equal(obs['prey4']['predator1'], np.array([-1, -3,  2]))
    np.testing.assert_array_equal(obs['prey4']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey4']['prey1'], np.array([1, -3,  1]))
    np.testing.assert_array_equal(obs['prey4']['prey2'], np.array([2, 0, 1]))
    np.testing.assert_array_equal(obs['prey4']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey4']['prey5'], np.array([1, -2, 1]))
    np.testing.assert_array_equal(obs['prey4']['prey6'], np.array([0, -4,  1]))

    np.testing.assert_array_equal(obs['prey5']['predator0'], np.array([-1, 1,  2]))
    np.testing.assert_array_equal(obs['prey5']['predator1'], np.array([-2, -1,  2]))
    np.testing.assert_array_equal(obs['prey5']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey5']['prey1'], np.array([0, -1,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey2'], np.array([1, 2,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey5']['prey4'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['prey5']['prey6'], np.array([-1, -2, 1]))

    np.testing.assert_array_equal(obs['prey6']['predator0'], np.array([0, 3,  2]))
    np.testing.assert_array_equal(obs['prey6']['predator1'], np.array([-1, 1,  2]))
    np.testing.assert_array_equal(obs['prey6']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey1'], np.array([1, 1,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey2'], np.array([2, 4,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey5'], np.array([1, 2, 1]))

    assert reward == {
        'predator0': 36,
        'predator1': -1,
        'prey1': -1,
        'prey2': -1,
        'prey4': -36,
        'prey5': -1,
        'prey6': -1,
    }

    assert done == {
        'predator0': False,
        'predator1': False,
        'prey1': False,
        'prey2': False,
        'prey4': True,
        'prey5': False,
        'prey6': False,
        '__all__': False}

    with pytest.raises(AssertionError):
        obs, reward, done, info = sim.step({
            'predator0': {'attack': 1, 'move': np.array([0, 0])},
            'predator1': {'attack': 1, 'move': np.array([0, 0])},
            'prey1': np.array([0, -1]),
            'prey2': np.array([1, 1]),
            'prey4': np.array([-1, 1]),
            'prey5': np.array([1, 1]),
            'prey6': np.array([0, 0]),
        })

    obs, reward, done, info = sim.step({
        'predator0': {'attack': 1, 'move': np.array([0, 0])},
        'predator1': {'attack': 1, 'move': np.array([0, 0])},
        'prey1': np.array([-1, 0]),
        'prey2': np.array([-1, 0]),
        'prey5': np.array([0, 1]),
        'prey6': np.array([-1, 0]),
    })

    np.testing.assert_array_equal(obs['predator0']['predator1'], np.array([-1, -2,  2]))
    np.testing.assert_array_equal(obs['predator0']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey1'], np.array([0, -2,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey2'], np.array([1, 1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey5'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey6'], np.array([0, 0,  0]))

    np.testing.assert_array_equal(obs['predator1']['predator0'], np.array([1, 2,  2]))
    np.testing.assert_array_equal(obs['predator1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey1'], np.array([1, 0,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey2'], np.array([2, 3, 1]))
    np.testing.assert_array_equal(obs['predator1']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey5'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey6'], np.array([0, 0,  0]))

    np.testing.assert_array_equal(obs['prey1']['predator0'], np.array([0, 2,  2]))
    np.testing.assert_array_equal(obs['prey1']['predator1'], np.array([-1, 0,  2]))
    np.testing.assert_array_equal(obs['prey1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey2'], np.array([1, 3, 1]))
    np.testing.assert_array_equal(obs['prey1']['prey3'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['prey1']['prey4'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['prey1']['prey5'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['prey1']['prey6'], np.array([0, 0, 0]))

    np.testing.assert_array_equal(obs['prey2']['predator0'], np.array([-1, -1,  2]))
    np.testing.assert_array_equal(obs['prey2']['predator1'], np.array([-2, -3,  2]))
    np.testing.assert_array_equal(obs['prey2']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey1'], np.array([-1, -3, 1]))
    np.testing.assert_array_equal(obs['prey2']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey5'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey6'], np.array([0, 0,  0]))

    np.testing.assert_array_equal(obs['prey5']['predator0'], np.array([-1, 1,  2]))
    np.testing.assert_array_equal(obs['prey5']['predator1'], np.array([-2, -1,  2]))
    np.testing.assert_array_equal(obs['prey5']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey5']['prey1'], np.array([-1, -1,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey2'], np.array([0, 2,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey5']['prey4'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['prey5']['prey6'], np.array([0, 0, 0]))

    np.testing.assert_array_equal(obs['prey6']['predator0'], np.array([0, 3,  2]))
    np.testing.assert_array_equal(obs['prey6']['predator1'], np.array([-1, 1,  2]))
    np.testing.assert_array_equal(obs['prey6']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey1'], np.array([0, 1,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey2'], np.array([1, 4,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey5'], np.array([0, 0, 0]))

    assert reward == {
        'predator0': 36,
        'predator1': 36,
        'prey1': -1,
        'prey2': -1,
        'prey5': -36,
        'prey6': -36
    }

    assert done == {
        'predator0': False,
        'predator1': False,
        'prey1': False,
        'prey2': False,
        'prey5': True,
        'prey6': True,
        '__all__': False}

    with pytest.raises(AssertionError):
        obs, reward, done, info = sim.step({
            'predator0': {'attack': 1, 'move': np.array([0, 0])},
            'predator1': {'attack': 1, 'move': np.array([0, 0])},
            'prey1': np.array([0, -1]),
            'prey2': np.array([1, 1]),
            'prey5': np.array([1, 1]),
            'prey6': np.array([0, 0]),
        })

    obs, reward, done, info = sim.step({
        'predator0': {'attack': 1, 'move': np.array([0, 0])},
        'predator1': {'attack': 1, 'move': np.array([0, 0])},
        'prey1': np.array([-1, 0]),
        'prey2': np.array([-1, 0]),
    })

    np.testing.assert_array_equal(obs['predator0']['predator1'], np.array([-1, -2,  2]))
    np.testing.assert_array_equal(obs['predator0']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey1'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey2'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey5'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey6'], np.array([0, 0,  0]))

    np.testing.assert_array_equal(obs['predator1']['predator0'], np.array([1, 2,  2]))
    np.testing.assert_array_equal(obs['predator1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey1'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey2'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey5'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey6'], np.array([0, 0,  0]))

    np.testing.assert_array_equal(obs['prey1']['predator0'], np.array([0, 2,  2]))
    np.testing.assert_array_equal(obs['prey1']['predator1'], np.array([-1, 0,  2]))
    np.testing.assert_array_equal(obs['prey1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey2'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['prey1']['prey3'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['prey1']['prey4'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['prey1']['prey5'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['prey1']['prey6'], np.array([0, 0, 0]))

    np.testing.assert_array_equal(obs['prey2']['predator0'], np.array([-1, -1,  2]))
    np.testing.assert_array_equal(obs['prey2']['predator1'], np.array([-2, -3,  2]))
    np.testing.assert_array_equal(obs['prey2']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['prey2']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey5'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey6'], np.array([0, 0,  0]))

    assert reward == {
        'predator0': 36,
        'predator1': 36,
        'prey1': -36,
        'prey2': -36,
    }

    assert done == {
        'predator0': False,
        'predator1': False,
        'prey1': True,
        'prey2': True,
        '__all__': True}


def test_turn_based_predator_prey_grid():
    np.random.seed(24)
    predators = [Predator(id=f'predator{i}', attack=1, view=0) for i in range(2)]
    prey = [Prey(id=f'prey{i}', view=0) for i in range(7)]
    agents = predators + prey
    sim_config = {
        'region': 6,
        'observation_mode': PredatorPreySimulation.ObservationMode.GRID,
        'agents': agents,
    }
    sim = PredatorPreySimulation.build(sim_config)
    sim = AllStepManager(sim)

    # Little hackish here because I have to explicitly set their values
    obs = sim.reset()
    sim.agents['predator0'].position = np.array([2, 3])
    sim.agents['predator1'].position = np.array([0, 1])
    sim.agents['prey0'].position = np.array([1, 1])
    sim.agents['prey1'].position = np.array([4, 3])
    sim.agents['prey2'].position = np.array([4, 3])
    sim.agents['prey3'].position = np.array([2, 3])
    sim.agents['prey4'].position = np.array([3, 3])
    sim.agents['prey5'].position = np.array([3, 1])
    sim.agents['prey6'].position = np.array([2, 1])
    obs = {agent_id: sim.sim.get_obs(agent_id) for agent_id in sim.agents}

    assert 'predator0' in obs
    assert 'predator0' in obs
    assert 'prey0' in obs
    assert 'prey1' in obs
    assert 'prey2' in obs
    assert 'prey3' in obs
    assert 'prey4' in obs
    assert 'prey5' in obs
    assert 'prey6' in obs

    obs, reward, done, info = sim.step({
        'predator0': {'attack': 1, 'move': np.array([0, 0])},
        'predator1': {'attack': 1, 'move': np.array([0, 0])},
        'prey0': {'move': np.array([1, 1]), 'harvest': 0},
        'prey1': {'move': np.array([0, -1]), 'harvest': 0},
        'prey2': {'move': np.array([1, 1]), 'harvest': 0},
        'prey3': {'move': np.array([0, 0]), 'harvest': 0},
        'prey4': {'move': np.array([-1, 1]), 'harvest': 0},
        'prey5': {'move': np.array([1, 1]), 'harvest': 0},
        'prey6': {'move': np.array([0, 0]), 'harvest': 0},
    })

    assert 'predator0' in obs
    assert 'predator0' in obs
    assert 'prey0' in obs
    assert 'prey1' in obs
    assert 'prey2' in obs
    assert 'prey3' in obs
    assert 'prey4' in obs
    assert 'prey5' in obs
    assert 'prey6' in obs

    assert reward == {
        'predator0': 36,
        'predator1': 36,
        'prey0': -36,
        'prey1': -1,
        'prey2': -1,
        'prey3': -36,
        'prey4': -1,
        'prey5': -1,
        'prey6': 0,
    }

    assert done == {
        'predator0': False,
        'predator1': False,
        'prey0': True,
        'prey1': False,
        'prey2': False,
        'prey3': True,
        'prey4': False,
        'prey5': False,
        'prey6': False,
        '__all__': False}


    with pytest.raises(AssertionError):
        obs, reward, done, info = sim.step({
            'predator0': {'attack': 1, 'move': np.array([0, 0])},
            'predator1': {'attack': 1, 'move': np.array([0, 0])},
            'prey0': {'move': np.array([0, -1]), 'harvest': 0},
            'prey1': {'move': np.array([0, -1]), 'harvest': 0},
            'prey2': {'move': np.array([1, 1]), 'harvest': 0},
            'prey3': {'move': np.array([0, -1]), 'harvest': 0},
            'prey4': {'move': np.array([0, -1]), 'harvest': 0},
            'prey5': {'move': np.array([1, 1]), 'harvest': 0},
            'prey6': {'move': np.array([0, 0]), 'harvest': 0},
        })

    obs, reward, done, info = sim.step({
        'predator0': {'attack': 1, 'move': np.array([0, 0])},
        'predator1': {'attack': 0, 'move': np.array([1, 0])},
        'prey1': {'move': np.array([-1, -1]), 'harvest': 0},
        'prey2': {'move': np.array([-1, 0]), 'harvest': 0},
        'prey4': {'move': np.array([0, -1]), 'harvest': 0},
        'prey5': {'move': np.array([-1, 0]), 'harvest': 0},
        'prey6': {'move': np.array([0, -1]), 'harvest': 0},
    })

    assert 'predator0' in obs
    assert 'predator0' in obs
    assert 'prey1' in obs
    assert 'prey2' in obs
    assert 'prey4' in obs
    assert 'prey5' in obs
    assert 'prey6' in obs

    assert reward == {
        'predator0': 36,
        'predator1': -1,
        'prey1': -1,
        'prey2': -1,
        'prey4': -36,
        'prey5': -1,
        'prey6': -1,
    }

    assert done == {
        'predator0': False,
        'predator1': False,
        'prey1': False,
        'prey2': False,
        'prey4': True,
        'prey5': False,
        'prey6': False,
        '__all__': False}


    with pytest.raises(AssertionError):
        obs, reward, done, info = sim.step({
            'predator0': {'attack': 1, 'move': np.array([0, 0])},
            'predator1': {'attack': 1, 'move': np.array([0, 0])},
            'prey1': {'move': np.array([0, -1]), 'harvest': 0},
            'prey2': {'move': np.array([1, 1]), 'harvest': 0},
            'prey4': {'move': np.array([0, -1]), 'harvest': 0},
            'prey5': {'move': np.array([1, 1]), 'harvest': 0},
            'prey6': {'move': np.array([0, 0]), 'harvest': 0},
        })

    obs, reward, done, info = sim.step({
        'predator0': {'attack': 1, 'move': np.array([0, 0])},
        'predator1': {'attack': 1, 'move': np.array([0, 0])},
        'prey1': {'move': np.array([-1, 0]), 'harvest': 0},
        'prey2': {'move': np.array([-1, 0]), 'harvest': 0},
        'prey5': {'move': np.array([-1, 0]), 'harvest': 0},
        'prey6': {'move': np.array([1, -1]), 'harvest': 0},
    })

    assert 'predator0' in obs
    assert 'predator0' in obs
    assert 'prey1' in obs
    assert 'prey2' in obs
    assert 'prey5' in obs
    assert 'prey6' in obs

    assert reward == {
        'predator0': 36,
        'predator1': 36,
        'prey1': -1,
        'prey2': -1,
        'prey5': -36,
        'prey6': -36,
    }

    assert done == {
        'predator0': False,
        'predator1': False,
        'prey1': False,
        'prey2': False,
        'prey5': True,
        'prey6': True,
        '__all__': False}


    with pytest.raises(AssertionError):
        obs, reward, done, info = sim.step({
            'predator0': {'attack': 1, 'move': np.array([0, 0])},
            'predator1': {'attack': 1, 'move': np.array([0, 0])},
            'prey1': {'move': np.array([0, -1]), 'harvest': 0},
            'prey2': {'move': np.array([1, 1]), 'harvest': 0},
            'prey5': {'move': np.array([1, 1]), 'harvest': 0},
            'prey6': {'move': np.array([0, 0]), 'harvest': 0},
        })

    obs, reward, done, info = sim.step({
        'predator0': {'attack': 1, 'move': np.array([0, 0])},
        'predator1': {'attack': 1, 'move': np.array([0, 0])},
        'prey1': {'move': np.array([-1, 0]), 'harvest': 0},
        'prey2': {'move': np.array([-1, 0]), 'harvest': 0},
    })

    assert 'predator0' in obs
    assert 'predator0' in obs
    assert 'prey1' in obs
    assert 'prey2' in obs

    assert reward == {
        'predator0': 36,
        'predator1': 36,
        'prey1': -36,
        'prey2': -36,
    }

    assert done == {
        'predator0': False,
        'predator1': False,
        'prey1': True,
        'prey2': True,
        '__all__': True}
