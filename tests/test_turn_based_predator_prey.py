import numpy as np

from abmarl.sim.predator_prey import PredatorPreySimulation, Predator, Prey
from abmarl.managers import TurnBasedManager


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
    sim = TurnBasedManager(sim)

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
    obs = {'predator0': sim.sim.get_obs('predator0')}

    np.testing.assert_array_equal(obs['predator0']['predator1'], np.array([-2, -2,  2]))
    np.testing.assert_array_equal(obs['predator0']['prey0'], np.array([-1, -2,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey1'], np.array([2, 0,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey2'], np.array([2, 0,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey3'], np.array([0, 0,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey4'], np.array([1, 0,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey5'], np.array([1, -2,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey6'], np.array([0, -2,  1]))

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 1, 'move': np.array([0, 0])} for agent_id in obs}
    )
    np.testing.assert_array_equal(obs['predator1']['predator0'], np.array([2, 2,  2]))
    np.testing.assert_array_equal(obs['predator1']['prey0'], np.array([1, 0,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey1'], np.array([4, 2,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey2'], np.array([4, 2,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey4'], np.array([3, 2,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey5'], np.array([3, 0,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey6'], np.array([2, 0,  1]))
    assert reward == {'predator1': 0}
    assert done == {'predator1': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 1, 'move': np.array([0, 0])} for agent_id in obs}
    )
    np.testing.assert_array_equal(obs['prey0']['predator0'], np.array([1, 2,  2]))
    np.testing.assert_array_equal(obs['prey0']['predator1'], np.array([-1, 0,  2]))
    np.testing.assert_array_equal(obs['prey0']['prey1'], np.array([3, 2,  1]))
    np.testing.assert_array_equal(obs['prey0']['prey2'], np.array([3, 2,  1]))
    np.testing.assert_array_equal(obs['prey0']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey0']['prey4'], np.array([2, 2,  1]))
    np.testing.assert_array_equal(obs['prey0']['prey5'], np.array([2, 0,  1]))
    np.testing.assert_array_equal(obs['prey0']['prey6'], np.array([1, 0,  1]))
    np.testing.assert_array_equal(obs['prey1']['predator0'], np.array([-2, 0,  2]))
    np.testing.assert_array_equal(obs['prey1']['predator1'], np.array([-4, -2,  2]))
    np.testing.assert_array_equal(obs['prey1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey2'], np.array([0, 0,  1]))
    np.testing.assert_array_equal(obs['prey1']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey4'], np.array([-1, 0,  1]))
    np.testing.assert_array_equal(obs['prey1']['prey5'], np.array([-1, -2,  1]))
    np.testing.assert_array_equal(obs['prey1']['prey6'], np.array([-2, -2,  1]))
    assert reward == {'prey0': -36, 'prey1': 0}
    assert done == {'prey0': True, 'prey1': False, '__all__': False}

    obs, reward, done, info = sim.step({'prey1': np.array([0, -1])})
    np.testing.assert_array_equal(obs['prey2']['predator0'], np.array([-2, 0,  2]))
    np.testing.assert_array_equal(obs['prey2']['predator1'], np.array([-4, -2,  2]))
    np.testing.assert_array_equal(obs['prey2']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey1'], np.array([0, -1,  1]))
    np.testing.assert_array_equal(obs['prey2']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey4'], np.array([-1, 0,  1]))
    np.testing.assert_array_equal(obs['prey2']['prey5'], np.array([-1, -2,  1]))
    np.testing.assert_array_equal(obs['prey2']['prey6'], np.array([-2, -2,  1]))
    assert reward == {'prey2': 0}
    assert done == {'prey2': False, '__all__': False}

    obs, reward, done, info = sim.step({agent_id: np.array([1, 1]) for agent_id in obs})
    np.testing.assert_array_equal(obs['prey3']['predator0'], np.array([0, 0,  2]))
    np.testing.assert_array_equal(obs['prey3']['predator1'], np.array([-2, -2,  2]))
    np.testing.assert_array_equal(obs['prey3']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey3']['prey1'], np.array([2, -1,  1]))
    np.testing.assert_array_equal(obs['prey3']['prey2'], np.array([3, 1,  1]))
    np.testing.assert_array_equal(obs['prey3']['prey4'], np.array([1, 0,  1]))
    np.testing.assert_array_equal(obs['prey3']['prey5'], np.array([1, -2,  1]))
    np.testing.assert_array_equal(obs['prey3']['prey6'], np.array([0, -2,  1]))
    np.testing.assert_array_equal(obs['prey4']['predator0'], np.array([-1, 0,  2]))
    np.testing.assert_array_equal(obs['prey4']['predator1'], np.array([-3, -2,  2]))
    np.testing.assert_array_equal(obs['prey4']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey4']['prey1'], np.array([1, -1,  1]))
    np.testing.assert_array_equal(obs['prey4']['prey2'], np.array([2, 1,  1]))
    np.testing.assert_array_equal(obs['prey4']['prey3'], np.array([-0, 0,  0]))
    np.testing.assert_array_equal(obs['prey4']['prey5'], np.array([0, -2,  1]))
    np.testing.assert_array_equal(obs['prey4']['prey6'], np.array([-1, -2,  1]))
    assert reward == {'prey3': -36, 'prey4': 0}
    assert done == {'prey3': True, 'prey4': False, '__all__': False}

    obs, reward, done, info = sim.step({'prey4': np.array([-1, 1])})
    np.testing.assert_array_equal(obs['prey5']['predator0'], np.array([-1, 2,  2]))
    np.testing.assert_array_equal(obs['prey5']['predator1'], np.array([-3, 0,  2]))
    np.testing.assert_array_equal(obs['prey5']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey5']['prey1'], np.array([1, 1,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey2'], np.array([2, 3,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey5']['prey4'], np.array([-1, 3,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey6'], np.array([-1, 0,  1]))
    assert reward == {'prey5': 0}
    assert done == {'prey5': False, '__all__': False}

    obs, reward, done, info = sim.step({agent_id: np.array([1, 1]) for agent_id in obs})
    np.testing.assert_array_equal(obs['prey6']['predator0'], np.array([0, 2,  2]))
    np.testing.assert_array_equal(obs['prey6']['predator1'], np.array([-2, 0,  2]))
    np.testing.assert_array_equal(obs['prey6']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey1'], np.array([2, 1,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey2'], np.array([3, 3,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey4'], np.array([0, 3,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey5'], np.array([2, 1,  1]))
    assert reward == {'prey6': 0}
    assert done == {'prey6': False, '__all__': False}

    obs, reward, done, info = sim.step({agent_id: np.array([0, 0]) for agent_id in obs})
    np.testing.assert_array_equal(obs['predator0']['predator1'], np.array([-2, -2,  2]))
    np.testing.assert_array_equal(obs['predator0']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey1'], np.array([2, -1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey2'], np.array([3, 1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey4'], np.array([0, 1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey5'], np.array([2, -1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey6'], np.array([0, -2,  1]))
    assert reward == {'predator0':36}
    assert done == {'predator0': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 1, 'move': np.array([0, 0])} for agent_id in obs}
    )
    np.testing.assert_array_equal(obs['predator1']['predator0'], np.array([2, 2,  2]))
    np.testing.assert_array_equal(obs['predator1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey1'], np.array([4, 1,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey2'], np.array([5, 3,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey5'], np.array([4, 1,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey6'], np.array([2, 0,  1]))
    assert reward == {'predator1': 36}
    assert done == {'predator1': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 0, 'move': np.array([1, 0])} for agent_id in obs}
    )
    np.testing.assert_array_equal(obs['prey1']['predator0'], np.array([-2, 1,  2]))
    np.testing.assert_array_equal(obs['prey1']['predator1'], np.array([-3, -1,  2]))
    np.testing.assert_array_equal(obs['prey1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey2'], np.array([1, 2,  1]))
    np.testing.assert_array_equal(obs['prey1']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey5'], np.array([0, 0,  1]))
    np.testing.assert_array_equal(obs['prey1']['prey6'], np.array([-2, -1,  1]))
    assert reward == {'prey1': -1}
    assert done == {'prey1': False, '__all__': False}

    obs, reward, done, info = sim.step({agent_id: np.array([-1, -1]) for agent_id in obs})
    np.testing.assert_array_equal(obs['prey2']['predator0'], np.array([-3, -1,  2]))
    np.testing.assert_array_equal(obs['prey2']['predator1'], np.array([-4, -3,  2]))
    np.testing.assert_array_equal(obs['prey2']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey1'], np.array([-2, -3,  1]))
    np.testing.assert_array_equal(obs['prey2']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey5'], np.array([-1, -2,  1]))
    np.testing.assert_array_equal(obs['prey2']['prey6'], np.array([-3, -3,  1]))
    assert reward == {'prey2': -1}
    assert done == {'prey2': False, '__all__': False}

    obs, reward, done, info = sim.step({agent_id: np.array([-1, 0]) for agent_id in obs})
    np.testing.assert_array_equal(obs['prey4']['predator0'], np.array([0, -1,  2]))
    np.testing.assert_array_equal(obs['prey4']['predator1'], np.array([-1, -3,  2]))
    np.testing.assert_array_equal(obs['prey4']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey4']['prey1'], np.array([1, -3,  1]))
    np.testing.assert_array_equal(obs['prey4']['prey2'], np.array([2, 0,  1]))
    np.testing.assert_array_equal(obs['prey4']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey4']['prey5'], np.array([2, -2,  1]))
    np.testing.assert_array_equal(obs['prey4']['prey6'], np.array([0, -3,  1]))
    np.testing.assert_array_equal(obs['prey5']['predator0'], np.array([-2, 1,  2]))
    np.testing.assert_array_equal(obs['prey5']['predator1'], np.array([-3, -1,  2]))
    np.testing.assert_array_equal(obs['prey5']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey5']['prey1'], np.array([-1, -1,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey2'], np.array([0, 2,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey5']['prey4'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['prey5']['prey6'], np.array([-2, -1,  1]))
    assert reward == {'prey4': -37, 'prey5': -1}
    assert done == {'prey4': True, 'prey5': False, '__all__': False}

    obs, reward, done, info = sim.step({'prey5': np.array([-1, 0])})
    np.testing.assert_array_equal(obs['prey6']['predator0'], np.array([0, 2,  2]))
    np.testing.assert_array_equal(obs['prey6']['predator1'], np.array([-1, 0,  2]))
    np.testing.assert_array_equal(obs['prey6']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey1'], np.array([1, 0,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey2'], np.array([2, 3,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey5'], np.array([1, 1,  1]))
    assert reward == {'prey6': 0}
    assert done == {'prey6': False, '__all__': False}

    obs, reward, done, info = sim.step({agent_id: np.array([0, -1]) for agent_id in obs})
    np.testing.assert_array_equal(obs['predator0']['predator1'], np.array([-1, -2,  2]))
    np.testing.assert_array_equal(obs['predator0']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey1'], np.array([1, -2,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey2'], np.array([2, 1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey5'], np.array([1, -1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey6'], np.array([0, -3,  1]))
    assert reward == {'predator0': 36}
    assert done == {'predator0': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 1, 'move': np.array([0, 0])} for agent_id in obs}
    )
    np.testing.assert_array_equal(obs['predator1']['predator0'], np.array([1, 2,  2]))
    np.testing.assert_array_equal(obs['predator1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey1'], np.array([2, 0,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey2'], np.array([3, 3,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey5'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey6'], np.array([1, -1,  1]))
    assert reward == {'predator1': -1}
    assert done == {'predator1': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 1, 'move': np.array([0, 0])} for agent_id in obs}
    )
    np.testing.assert_array_equal(obs['prey1']['predator0'], np.array([-1, 2,  2]))
    np.testing.assert_array_equal(obs['prey1']['predator1'], np.array([-2, 0,  2]))
    np.testing.assert_array_equal(obs['prey1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey2'], np.array([1, 3,  1]))
    np.testing.assert_array_equal(obs['prey1']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey5'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey1']['prey6'], np.array([0, 0,  0]))
    assert reward == {'prey1': -1}
    assert done == {'prey1': False, '__all__': False}

    obs, reward, done, info = sim.step({agent_id: np.array([-1, 0]) for agent_id in obs})
    np.testing.assert_array_equal(obs['prey2']['predator0'], np.array([-2, -1,  2]))
    np.testing.assert_array_equal(obs['prey2']['predator1'], np.array([-3, -3,  2]))
    np.testing.assert_array_equal(obs['prey2']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey1'], np.array([-2, -3,  1]))
    np.testing.assert_array_equal(obs['prey2']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey5'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey6'], np.array([0, 0,  0]))
    assert reward == {'prey2': -1}
    assert done == {'prey2': False, '__all__': False}

    obs, reward, done, info = sim.step({agent_id: np.array([-1, 0]) for agent_id in obs})
    np.testing.assert_array_equal(obs['prey5']['predator0'], np.array([-1, 1,  2]))
    np.testing.assert_array_equal(obs['prey5']['predator1'], np.array([-2, -1,  2]))
    np.testing.assert_array_equal(obs['prey5']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey5']['prey1'], np.array([-1, -1,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey2'], np.array([0, 2,  1]))
    np.testing.assert_array_equal(obs['prey5']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey5']['prey4'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(obs['prey5']['prey6'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['predator0'], np.array([0, 3,  2]))
    np.testing.assert_array_equal(obs['prey6']['predator1'], np.array([-1, 1,  2]))
    np.testing.assert_array_equal(obs['prey6']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey1'], np.array([0, 1,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey2'], np.array([1, 4,  1]))
    np.testing.assert_array_equal(obs['prey6']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey6']['prey5'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['predator1'], np.array([-1, -2,  2]))
    np.testing.assert_array_equal(obs['predator0']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey1'], np.array([0, -2,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey2'], np.array([1, 1,  1]))
    np.testing.assert_array_equal(obs['predator0']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey5'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator0']['prey6'], np.array([0, 0,  0]))
    assert reward == {'prey5': -37, 'prey6': -37, 'predator0': 36}
    assert done == {'prey5': True, 'prey6': True, 'predator0': False, '__all__': False}

    obs, reward, done, info = sim.step({'predator0': {'attack': 1, 'move': np.array([0, 0])}})
    np.testing.assert_array_equal(obs['predator1']['predator0'], np.array([1, 2,  2]))
    np.testing.assert_array_equal(obs['predator1']['prey0'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey1'], np.array([1, 0,  1]))
    np.testing.assert_array_equal(obs['predator1']['prey2'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey5'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['predator1']['prey6'], np.array([0, 0,  0]))
    assert reward == {'predator1': 36}
    assert done == {'predator1': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 1, 'move': np.array([0, 0])} for agent_id in obs}
    )
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
    np.testing.assert_array_equal(obs['prey2']['prey1'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey3'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey4'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey5'], np.array([0, 0,  0]))
    np.testing.assert_array_equal(obs['prey2']['prey6'], np.array([0, 0,  0]))
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
    assert reward == {'prey1': -37, 'prey2': -37, 'predator0': 36, 'predator1': 36}
    assert done == {
        'prey1': True, 'prey2': True, 'predator0': False, 'predator1': False, '__all__': True
    }


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
    sim = TurnBasedManager(sim)

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
    obs = {'predator0': sim.sim.get_obs('predator0')}

    assert len(obs) == 1 and 'predator0' in obs

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 1, 'move': np.array([0, 0])} for agent_id in obs}
    )
    assert len(obs) == 1 and 'predator1' in obs
    assert reward == {'predator1': 0}
    assert done == {'predator1': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 1, 'move': np.array([0, 0])} for agent_id in obs}
    )
    assert len(obs) == 2 and 'prey0' in obs and 'prey1' in obs
    assert reward == {'prey0': -36, 'prey1': 0}
    assert done == {'prey0': True, 'prey1': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {'prey1': {'move': np.array([0, -1]), 'harvest': 0}}
    )
    assert len(obs) == 1 and 'prey2' in obs
    assert reward == {'prey2': 0}
    assert done == {'prey2': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'move': np.array([1, 1]), 'harvest': 0} for agent_id in obs}
    )
    assert len(obs) == 2 and 'prey3' in obs and 'prey4' in obs
    assert reward == {'prey3': -36, 'prey4': 0}
    assert done == {'prey3': True, 'prey4': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {'prey4': {'move': np.array([-1, 1]), 'harvest': 0} for agent_id in obs}
    )
    assert len(obs) == 1 and 'prey5' in obs
    assert reward == {'prey5': 0}
    assert done == {'prey5': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'move': np.array([1, 1]), 'harvest': 0} for agent_id in obs}
    )
    assert len(obs) == 1 and 'prey6' in obs
    assert reward == {'prey6': 0}
    assert done == {'prey6': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'move': np.array([0, 0]), 'harvest': 0} for agent_id in obs}
    )
    assert len(obs) == 1 and 'predator0' in obs
    assert reward == {'predator0':36}
    assert done == {'predator0': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 1, 'move': np.array([0, 0])} for agent_id in obs}
    )
    assert len(obs) == 1 and 'predator1' in obs
    assert reward == {'predator1': 36}
    assert done == {'predator1': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 0, 'move': np.array([1, 0])} for agent_id in obs}
    )
    assert len(obs) == 1 and 'prey1' in obs
    assert reward == {'prey1': -1}
    assert done == {'prey1': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'move': np.array([-1, -1]), 'harvest': 0} for agent_id in obs}
    )
    assert len(obs) == 1 and 'prey2' in obs
    assert reward == {'prey2': -1}
    assert done == {'prey2': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'move': np.array([-1, 0]), 'harvest': 0} for agent_id in obs}
    )
    assert len(obs) == 2 and 'prey4' in obs and 'prey5'
    assert reward == {'prey4': -37, 'prey5': -1}
    assert done == {'prey4': True, 'prey5': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {'prey5': {'move': np.array([-1, 0]), 'harvest': 0} for agent_id in obs}
    )
    assert len(obs) == 1 and 'prey6' in obs
    assert reward == {'prey6': 0}
    assert done == {'prey6': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'move': np.array([0, -1]), 'harvest': 0} for agent_id in obs}
    )
    assert len(obs) == 1 and 'predator0' in obs
    assert reward == {'predator0': 36}
    assert done == {'predator0': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 1, 'move': np.array([0, 0])} for agent_id in obs}
    )
    assert len(obs) == 1 and 'predator1' in obs
    assert reward == {'predator1': -1}
    assert done == {'predator1': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 1, 'move': np.array([0, 0])} for agent_id in obs}
    )
    assert len(obs) == 1 and 'prey1' in obs
    assert reward == {'prey1': -1}
    assert done == {'prey1': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'move': np.array([-1, 0]), 'harvest': 0} for agent_id in obs}
    )
    assert len(obs) == 1 and 'prey2' in obs
    assert reward == {'prey2': -1}
    assert done == {'prey2': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'move': np.array([-1, 0]), 'harvest': 0} for agent_id in obs}
    )
    assert len(obs) == 3 and 'prey5' in obs and 'prey6' in obs and 'predator0' in obs
    assert reward == {'prey5': -37, 'prey6': -37, 'predator0': 36}
    assert done == {'prey5': True, 'prey6': True, 'predator0': False, '__all__': False}

    obs, reward, done, info = sim.step({'predator0': {'attack': 1, 'move': np.array([0, 0])}})
    assert len(obs) == 1 and 'predator1' in obs
    assert reward == {'predator1': 36}
    assert done == {'predator1': False, '__all__': False}

    obs, reward, done, info = sim.step(
        {agent_id: {'attack': 1, 'move': np.array([0, 0])} for agent_id in obs}
    )
    assert len(obs) == 4
    assert 'prey1' in obs
    assert 'prey2' in obs
    assert 'predator0' in obs
    assert 'predator1' in obs
    assert reward == {'prey1': -37, 'prey2': -37, 'predator0': 36, 'predator1': 36}
    assert done == {
        'prey1': True, 'prey2': True, 'predator0': False, 'predator1': False, '__all__': True
    }
