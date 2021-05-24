
from gym.spaces import Dict, Discrete, Box
import numpy as np
import pytest

from admiral.envs.examples.predator_prey import PredatorPreyEnv, PredatorPreyEnvDistanceObs, PredatorPreyEnvGridObs
from admiral.envs.examples.predator_prey import Predator, Prey
from admiral.envs.examples.predator_prey.grid_resources import GridResources

def test_build_fails():
    with pytest.raises(TypeError): # Abstract class error
        PredatorPreyEnv()
    with pytest.raises(TypeError): # Missing argument
        PredatorPreyEnvDistanceObs()
    with pytest.raises(TypeError): # Missing argument
        PredatorPreyEnvGridObs()
    with pytest.raises(TypeError): # Abstract class error
        PredatorPreyEnv({
            'region': None,
            'max_steps': None,
            'agents': None,
            'reward_map': None,
        })

def test_class_attributes():
    assert PredatorPreyEnv.ActionStatus.BAD_MOVE == 0
    assert PredatorPreyEnv.ActionStatus.GOOD_MOVE == 1
    assert PredatorPreyEnv.ActionStatus.NO_MOVE == 2
    assert PredatorPreyEnv.ActionStatus.BAD_ATTACK == 3
    assert PredatorPreyEnv.ActionStatus.GOOD_ATTACK == 4
    assert PredatorPreyEnv.ActionStatus.EATEN == 5

    assert PredatorPreyEnv.ObservationMode.GRID == 0
    assert PredatorPreyEnv.ObservationMode.DISTANCE == 1

def test_builder():
    env = PredatorPreyEnv.build()
    assert isinstance(env, PredatorPreyEnvGridObs)
    assert env.region == 10
    assert env.max_steps == 200
    assert env.reward_map == {
        'predator': {
            PredatorPreyEnv.ActionStatus.BAD_MOVE: -10,
            PredatorPreyEnv.ActionStatus.GOOD_MOVE: -1,
            PredatorPreyEnv.ActionStatus.NO_MOVE: 0,
            PredatorPreyEnv.ActionStatus.BAD_ATTACK: -10,
            PredatorPreyEnv.ActionStatus.GOOD_ATTACK: 100,
        },
        'prey': {
            PredatorPreyEnv.ActionStatus.BAD_MOVE: -10,
            PredatorPreyEnv.ActionStatus.GOOD_MOVE: -1,
            PredatorPreyEnv.ActionStatus.NO_MOVE: 0,
            PredatorPreyEnv.ActionStatus.EATEN: -100,
            PredatorPreyEnv.ActionStatus.BAD_HARVEST: -10,
            PredatorPreyEnv.ActionStatus.GOOD_HARVEST: 10,
        }
    }
    grid_resources = GridResources.build({'region': env.region})
    assert env.resources.region == grid_resources.region
    assert env.resources.coverage == grid_resources.coverage
    assert env.resources.min_value == grid_resources.min_value
    assert env.resources.max_value == grid_resources.max_value
    assert env.resources.revive_rate == grid_resources.revive_rate

    agents = env.agents
    assert type(agents) == dict
    assert len(agents) == 2
    assert agents['prey0'].id == 'prey0'
    assert type(agents['prey0']) == Prey
    assert agents['prey0'].view == 9
    assert agents['prey0'].harvest_amount == 0.1
    assert agents['prey0'].observation_space == Dict({
        'agents': Box(low=-1, high=2, shape=(19,19), dtype=np.int),
        'resources': Box(-1, env.resources.max_value, (19,19), np.float),
    })
    assert agents['prey0'].action_space == Dict({
        'move': Box(low=-1.5, high=1.5, shape=(2,)),
        'harvest': Discrete(2)
    })
    assert agents['predator0'].id == 'predator0'
    assert type(agents['predator0']) == Predator
    assert agents['predator0'].view == 9
    assert agents['predator0'].attack == 0
    assert agents['predator0'].observation_space == Dict({
        'agents': Box(low=-1, high=2, shape=(19,19), dtype=np.int),
        'resources': Box(-1, env.resources.max_value, (19,19), np.float),
    })
    assert agents['predator0'].action_space == Dict({
        'move': Box(low=-1.5, high=1.5, shape=(2,)),
        'attack': Discrete(2)
    })

def test_builder_region():
    env = PredatorPreyEnv.build({'region': 20})
    assert env.region == 20
    assert env.resources.region == 20
    with pytest.raises(TypeError):
        PredatorPreyEnv.build({'region': '12'})
    with pytest.raises(TypeError):
        PredatorPreyEnv.build({'region': -2})

    agents = env.agents
    assert len(agents) == 2
    assert agents['prey0'].id == 'prey0'
    assert type(agents['prey0']) == Prey
    assert agents['prey0'].view == 19
    assert agents['prey0'].observation_space == Dict({
        'agents': Box(low=-1, high=2, shape=(39,39), dtype=np.int),
        'resources': Box(-1, env.resources.max_value, (39,39), np.float),
    })
    assert agents['predator0'].id == 'predator0'
    assert type(agents['predator0']) == Predator
    assert agents['predator0'].view == 19
    assert agents['predator0'].attack == 0
    assert agents['predator0'].observation_space == Dict({
        'agents': Box(low=-1, high=2, shape=(39,39), dtype=np.int),
        'resources': Box(-1, env.resources.max_value, (39,39), np.float),
    })

def test_build_max_steps():
    env = PredatorPreyEnv.build({'max_steps': 100})
    assert env.max_steps == 100
    with pytest.raises(TypeError):
        PredatorPreyEnv.build({'max_steps': 12.5})
    with pytest.raises(TypeError):
        PredatorPreyEnv.build({'max_steps': -8})

def test_builder_observation_mode():
    env = PredatorPreyEnv.build({'observation_mode': PredatorPreyEnv.ObservationMode.DISTANCE})
    assert isinstance(env, PredatorPreyEnvDistanceObs)

    agents = env.agents
    assert type(agents) == dict
    assert len(agents) == 2
    assert agents['prey0'].id == 'prey0'
    assert type(agents['prey0']) == Prey
    assert agents['prey0'].view == 9
    assert agents['prey0'].observation_space == Dict({
        'predator0': Box(-9, 9, (3,), np.int)
    })
    assert agents['predator0'].id == 'predator0'
    assert type(agents['predator0']) == Predator
    assert agents['predator0'].view == 9
    assert agents['predator0'].attack == 0
    assert agents['predator0'].observation_space == Dict({
        'prey0': Box(low=-9, high=9, shape=(3,), dtype=np.int)
    })

def test_builder_rewards():
    rewards = {
        'predator': {
            PredatorPreyEnv.ActionStatus.BAD_MOVE: -2, 
            PredatorPreyEnv.ActionStatus.GOOD_MOVE: -1, 
            PredatorPreyEnv.ActionStatus.NO_MOVE: 0,
            PredatorPreyEnv.ActionStatus.BAD_ATTACK: -5,
            PredatorPreyEnv.ActionStatus.GOOD_ATTACK: 10,
        },
        'prey': {
            PredatorPreyEnv.ActionStatus.BAD_MOVE: -2, 
            PredatorPreyEnv.ActionStatus.GOOD_MOVE: 2, 
            PredatorPreyEnv.ActionStatus.NO_MOVE: 1,
            PredatorPreyEnv.ActionStatus.EATEN: -10,
        }
    }
    env = PredatorPreyEnv.build({'rewards': rewards})
    assert env.reward_map == rewards
    with pytest.raises(TypeError):
        PredatorPreyEnv.build({'rewards': 12})

def test_builder_resources():
    resources = {
        'coverage': 0.5,
        'min_value': 0.3,
        'max_value': 1.2,
        'revive_rate': 0.1,
    }
    env = PredatorPreyEnv.build({'resources': resources})
    assert env.resources.region == env.region
    assert env.resources.coverage == resources['coverage']
    assert env.resources.min_value == resources['min_value']
    assert env.resources.max_value == resources['max_value']
    assert env.resources.revive_rate == resources['revive_rate']

def test_builder_agents():
    np.random.seed(24)
    # Create some good agents
    agents = [
        Prey(id='prey0', view=7, move=2),
        Predator(id='predator1', view=3, attack=2),
        Prey(id='prey2', view=5, move=3),
        Predator(id='predator3', view=2, move=2, attack=1),
        Predator(id='predator4', view=0, attack=3)
    ]
    env = PredatorPreyEnv.build({'agents': agents})

    agents = env.agents
    for agent in agents.values():
        assert agent.configured
        
    assert agents['prey0'].observation_space == Dict({
        'agents': Box(-1, 2, (15,15), np.int),
        'resources': Box(-1, env.resources.max_value, (15,15), np.float),
    })
    assert agents['predator1'].observation_space == Dict({
        'agents': Box(-1, 2, (7, 7), np.int),
        'resources': Box(-1, env.resources.max_value, (7,7), np.float),
    })
    assert agents['prey2'].observation_space == Dict({
        'agents': Box(-1, 2, (11, 11), np.int),
        'resources': Box(-1, env.resources.max_value, (11,11), np.float),
    })
    assert agents['predator3'].observation_space == Dict({
        'agents': Box(-1, 2, (5, 5), np.int),
        'resources': Box(-1, env.resources.max_value, (5,5), np.float),
    })
    assert agents['predator4'].observation_space == Dict({
        'agents': Box(-1, 2, (1, 1), np.int),
        'resources': Box(-1, env.resources.max_value, (1,1), np.float),
    })

    assert agents['prey0'].action_space == Dict({
        'move': Box(-2.5, 2.5, (2,)),
        'harvest': Discrete(2),
    })
    assert agents['predator1'].action_space == Dict({
        'attack': Discrete(2),
        'move': Box(-1.5, 1.5, (2,)),
    })
    assert agents['prey2'].action_space == Dict({
        'move': Box(-3.5, 3.5, (2,)),
        'harvest': Discrete(2),
    })
    assert agents['predator3'].action_space == Dict({
        'attack': Discrete(2),
        'move': Box(-2.5, 2.5, (2,)),
    })
    assert agents['predator4'].action_space == Dict({
        'attack': Discrete(2),
        'move': Box(-1.5, 1.5, (2,)),
    })

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
    
def test_reset_distance_obs():
    np.random.seed(24)
    agents = [
        Prey(id='prey0', view=2),
        Predator(id='predator1', view=4),
        Prey(id='prey2', view=2),
        Predator(id='predator3', view=4),
        Predator(id='predator4', view=4),
    ]
    env = PredatorPreyEnv.build({'agents': agents, 'observation_mode': PredatorPreyEnv.ObservationMode.DISTANCE})
    env.reset()

    # Explicitly place the agents
    env.agents['predator1'].position = np.array([4,4])
    env.agents['predator3'].position = np.array([3,3])
    env.agents['predator4'].position = np.array([7,9])
    env.agents['prey0'].position = np.array([1,1])
    env.agents['prey2'].position = np.array([3,2])

    assert env.step_count == 0

    np.testing.assert_array_equal(env.get_obs('predator1')['predator3'], np.array([-1, -1, 2]))
    np.testing.assert_array_equal(env.get_obs('predator1')['predator4'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator1')['prey0'], np.array([-3, -3, 1]))
    np.testing.assert_array_equal(env.get_obs('predator1')['prey2'], np.array([-1, -2, 1]))

    np.testing.assert_array_equal(env.get_obs('predator3')['predator1'], np.array([1, 1, 2]))
    np.testing.assert_array_equal(env.get_obs('predator3')['predator4'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator3')['prey0'], np.array([-2, -2, 1]))
    np.testing.assert_array_equal(env.get_obs('predator3')['prey2'], np.array([0, -1, 1]))

    np.testing.assert_array_equal(env.get_obs('predator4')['predator1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator4')['predator3'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator4')['prey0'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('predator4')['prey2'], np.array([0, 0, 0]))
    
    np.testing.assert_array_equal(env.get_obs('prey0')['predator1'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('prey0')['predator3'], np.array([2, 2, 2]))
    np.testing.assert_array_equal(env.get_obs('prey0')['predator4'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('prey0')['prey2'], np.array([2, 1, 1]))
    
    np.testing.assert_array_equal(env.get_obs('prey2')['predator1'], np.array([1, 2, 2]))
    np.testing.assert_array_equal(env.get_obs('prey2')['predator3'], np.array([0, 1, 2]))
    np.testing.assert_array_equal(env.get_obs('prey2')['predator4'], np.array([0, 0, 0]))
    np.testing.assert_array_equal(env.get_obs('prey2')['prey0'], np.array([-2, -1, 1]))

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

def test_attack_distances():
    region = 5
    predators = [Predator(id='predator0')]
    # predators = [{'id': 'predator0', 'view': region-1, 'move': 1, 'attack': 0}]
    prey = [Prey(id='prey{}'.format(i)) for i in range(3)]
    # prey = [{'id': 'prey' + str(i), 'view': region-1, 'move': 1} for i in range(3)]
    agents = predators + prey
    config = {'region': region, 'agents': agents}
    env = PredatorPreyEnv.build(config)
    env.reset()
    env.agents['predator0'].position = np.array([2, 2])
    env.agents['prey0'].position = np.array([2, 2])
    env.agents['prey1'].position = np.array([1, 1])
    env.agents['prey2'].position = np.array([0, 0])
    assert env.agents['predator0'].attack == 0
    action_dict = {
        'predator0': {'move': np.zeros([0, 0]), 'attack': 1},
        'prey0': {'move': np.zeros(2), 'harvest': 0},
        'prey1': {'move': np.zeros(2), 'harvest': 0},
        'prey2': {'move': np.zeros(2), 'harvest': 0},
    }


    env.step(action_dict)
    assert env.get_reward('predator0') == env.reward_map['predator'][PredatorPreyEnv.ActionStatus.GOOD_ATTACK]
    assert env.get_reward('prey0') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.EATEN]
    assert env.get_reward('prey1') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.NO_MOVE]
    assert env.get_reward('prey2') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.NO_MOVE]
    assert env.get_done('predator0') == False
    assert env.get_done('prey0') == True
    assert env.get_done('prey1') == False
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False
    
    del action_dict['prey0']
    env.step(action_dict)
    assert env.get_reward('predator0') == env.reward_map['predator'][PredatorPreyEnv.ActionStatus.BAD_ATTACK]
    assert env.get_reward('prey1') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.NO_MOVE]
    assert env.get_reward('prey2') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.NO_MOVE]
    assert env.get_done('predator0') == False
    assert env.get_done('prey1') == False
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False


    env.agents['predator0'].attack = 1

    env.step(action_dict)
    assert env.get_reward('predator0') == env.reward_map['predator'][PredatorPreyEnv.ActionStatus.GOOD_ATTACK]
    assert env.get_reward('prey1') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.EATEN]
    assert env.get_reward('prey2') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.NO_MOVE]
    assert env.get_done('predator0') == False
    assert env.get_done('prey1') == True
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False

    del action_dict['prey1']
    env.step(action_dict)
    assert env.get_reward('predator0') == env.reward_map['predator'][PredatorPreyEnv.ActionStatus.BAD_ATTACK]
    assert env.get_reward('prey2') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.NO_MOVE]
    assert env.get_done('predator0') == False
    assert env.get_done('prey2') == False
    assert env.get_all_done() == False


    env.agents['predator0'].attack = 2
    env.step(action_dict)
    assert env.get_reward('predator0') == env.reward_map['predator'][PredatorPreyEnv.ActionStatus.GOOD_ATTACK]
    assert env.get_reward('prey2') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.EATEN]
    assert env.get_done('predator0') == False
    assert env.get_done('prey2') == True
    assert env.get_all_done() == True

def test_diagonal_moves():
    np.random.seed(24)
    region = 3
    prey = [Prey(id='prey{}'.format(i)) for i in range(12)]
    env = PredatorPreyEnv.build({'agents': prey, 'region': region})
    env.reset()
    for prey in env.agents.values():
        prey.position = np.array([1,1])


    action = {
        'prey0': {'move': np.array([0, 1]), 'harvest': 0},
        'prey1': {'move': np.array([0, 1]), 'harvest': 0},
        'prey2': {'move': np.array([0, -1]), 'harvest': 0},
        'prey3': {'move': np.array([0, -1]), 'harvest': 0},
        'prey4': {'move': np.array([-1, 0]), 'harvest': 0},
        'prey5': {'move': np.array([-1, 0]), 'harvest': 0},
        'prey6': {'move': np.array([1, 0]), 'harvest': 0},
        'prey7': {'move': np.array([1, 0]), 'harvest': 0},
        'prey8': {'move': np.array([1, 1]), 'harvest': 0},
        'prey9': {'move': np.array([1, -1]), 'harvest': 0},
        'prey10': {'move': np.array([-1, -1]), 'harvest': 0},
        'prey11': {'move': np.array([-1, 1]), 'harvest': 0},
    }
    env.step(action)
    assert env.get_reward('prey0') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.GOOD_MOVE]
    assert env.get_reward('prey1') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.GOOD_MOVE]
    assert env.get_reward('prey2') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.GOOD_MOVE]
    assert env.get_reward('prey3') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.GOOD_MOVE]
    assert env.get_reward('prey4') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.GOOD_MOVE]
    assert env.get_reward('prey5') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.GOOD_MOVE]
    assert env.get_reward('prey6') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.GOOD_MOVE]
    assert env.get_reward('prey7') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.GOOD_MOVE]
    assert env.get_reward('prey8') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.GOOD_MOVE]
    assert env.get_reward('prey9') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.GOOD_MOVE]
    assert env.get_reward('prey10') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.GOOD_MOVE]
    assert env.get_reward('prey11') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.GOOD_MOVE]

    np.testing.assert_array_equal([agent.position for agent in env.agents.values()], [
        np.array([1, 2]),
        np.array([1, 2]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([2, 1]),
        np.array([2, 1]),
        np.array([2, 2]),
        np.array([2, 0]),
        np.array([0, 0]),
        np.array([0, 2]),
    ])


    action = {
        'prey0': {'move': np.array([1, 1]), 'harvest': 0},
        'prey1': {'move': np.array([-1, 1]), 'harvest': 0},
        'prey2': {'move': np.array([1, -1]), 'harvest': 0},
        'prey3': {'move': np.array([-1, -1]), 'harvest': 0},
        'prey4': {'move': np.array([-1, 1]), 'harvest': 0},
        'prey5': {'move': np.array([-1, -1]), 'harvest': 0},
        'prey6': {'move': np.array([1, 1]), 'harvest': 0},
        'prey7': {'move': np.array([1, -1]), 'harvest': 0},
        'prey8': {'move': np.array([1, 1]), 'harvest': 0},
        'prey9': {'move': np.array([1, -1]), 'harvest': 0},
        'prey10': {'move': np.array([-1, -1]), 'harvest': 0},
        'prey11': {'move': np.array([-1, 1]), 'harvest': 0},
    }
    env.step(action)
    assert env.get_reward('prey0') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.BAD_MOVE]
    assert env.get_reward('prey1') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.BAD_MOVE]
    assert env.get_reward('prey2') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.BAD_MOVE]
    assert env.get_reward('prey3') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.BAD_MOVE]
    assert env.get_reward('prey4') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.BAD_MOVE]
    assert env.get_reward('prey5') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.BAD_MOVE]
    assert env.get_reward('prey6') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.BAD_MOVE]
    assert env.get_reward('prey7') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.BAD_MOVE]
    assert env.get_reward('prey8') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.BAD_MOVE]
    assert env.get_reward('prey9') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.BAD_MOVE]
    assert env.get_reward('prey10') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.BAD_MOVE]
    assert env.get_reward('prey11') == env.reward_map['prey'][PredatorPreyEnv.ActionStatus.BAD_MOVE]

    np.testing.assert_array_equal([agent.position for agent in env.agents.values()], [
        np.array([1, 2]),
        np.array([1, 2]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([2, 1]),
        np.array([2, 1]),
        np.array([2, 2]),
        np.array([2, 0]),
        np.array([0, 0]),
        np.array([0, 2]),
    ])
    
def test_multi_move():
    np.random.seed(24)
    region = 8
    prey = [Prey(id='prey{}'.format(i), move=4) for i in range(4)]
    env = PredatorPreyEnv.build({'agents': prey, 'region': region})
    env.reset()
    env.agents['prey0'].position = np.array([2, 3])
    env.agents['prey1'].position = np.array([0, 7])
    env.agents['prey2'].position = np.array([1, 1])
    env.agents['prey3'].position = np.array([1, 4])

    action = {agent_id: agent.action_space.sample() for agent_id, agent in env.agents.items()}
    action = {
        'prey0': {'move': np.array([-2, 3]), 'harvest': 0},
        'prey1': {'move': np.array([4, 0]), 'harvest': 0},
        'prey2': {'move': np.array([2, 1]), 'harvest': 0},
        'prey3': {'move': np.array([3, -2]), 'harvest': 0},
    }
    env.step(action)

    np.testing.assert_array_equal(env.agents['prey0'].position, [0,6])
    np.testing.assert_array_equal(env.agents['prey1'].position, [4,7])
    np.testing.assert_array_equal(env.agents['prey2'].position, [3,2])
    np.testing.assert_array_equal(env.agents['prey3'].position, [4,2])
    
def test_done_on_max_steps():
    agents = [Prey(id=f'prey{i}') for i in range(2)]
    env = PredatorPreyEnv.build({'max_steps': 4, 'agents': agents})
    env.reset()
    for i in range(4):
        env.step({agent_id: agent.action_space.sample() for agent_id, agent in env.agents.items()})
    assert env.get_all_done() == True

def test_with_resources():
    np.random.seed(24)
    agents = [
        Prey(id='prey0', view=2, harvest_amount=0.3),
        Prey(id='prey1'),
        Predator(id='predator0', view=1)
    ]
    env = PredatorPreyEnv.build({'region': 5, 'agents': agents})
    env.reset()
    
    np.allclose(env.get_obs('predator0')['resources'], np.array([
        [0.        , 0.19804811, 0.        ],
        [0.16341112, 0.58086431, 0.4482749 ],
        [0.        , 0.38637824, 0.78831386]
    ]))
    np.allclose(env.get_obs('prey0')['resources'], np.array([
        [ 0.19804811, 0.        ,  0.42549817,  0.9438245 , -1.        ],
        [ 0.58086431,  0.4482749 ,  0.40239527,  0.31349653, -1.        ],
        [ 0.38637824,  0.78831386,  0.33666274,  0.71590738, -1.        ],
        [ 0.6264872 ,  0.65159097,  0.84080142,  0.24749604, -1.        ],
        [ 0.86455522,  0.        ,  0.        ,  0.        , -1.        ]]))
    np.allclose(env.get_obs('prey1')['resources'], np.array([
        [-1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        ],
        [-1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        ],
        [-1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        ],
        [-1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        ],
        [-1.        , -1.        , -1.        ,  0.        ,  0.19804811,  0.        ,  0.42549817,  0.9438245 , -1.        ],
        [-1.        , -1.        , -1.        ,  0.16341112,  0.58086431, 0.4482749 ,  0.40239527,  0.31349653, -1.        ],
        [-1.        , -1.        , -1.        ,  0.        ,  0.38637824, 0.78831386,  0.33666274, 0.71590738, -1.        ],
        [-1.        , -1.        , -1.        ,  0.        ,  0.6264872 , 0.65159097,  0.84080142,  0.24749604, -1.        ],
        [-1.        , -1.        , -1.        ,  0.67319672,  0.86455522, 0.        ,  0.        ,  0.        , -1.        ]
    ]))
    env.step({
        'prey0': {'move': np.zeros, 'harvest': 1},
        'prey1': {'move': np.zeros, 'harvest': 1},
    })
    
    np.allclose(env.get_obs('predator0')['resources'], np.array([
        [0.        , 0.19804811, 0.        ],
        [0.16341112, 0.58086431, 0.4482749 ],
        [0.        , 0.38637824, 0.78831386]
    ]))
    np.allclose(env.get_obs('prey0')['resources'], np.array([
        [ 0.19804811,  0.        ,  0.42549817,  0.9438245 , -1.        ],
        [ 0.58086431,  0.4482749 ,  0.40239527,  0.31349653, -1.        ],
        [ 0.38637824,  0.78831386,  0.07666274,  0.71590738, -1.        ],
        [ 0.6264872 ,  0.65159097,  0.84080142,  0.24749604, -1.        ],
        [ 0.86455522,  0.        ,  0.        ,  0.        , -1.        ]]))
    np.allclose(env.get_obs('prey1')['resources'], np.array([
        [-1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        ],
        [-1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        ],
        [-1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        ],
        [-1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        ],
        [-1.        , -1.        , -1.        ,  0.        ,  0.19804811, 0.        ,  0.42549817,  0.9438245 , -1.        ],
        [-1.        , -1.        , -1.        ,  0.16341112,  0.58086431, 0.4482749 ,  0.40239527,  0.31349653, -1.        ],
        [-1.        , -1.        , -1.        ,  0.        ,  0.38637824, 0.78831386,  0.07666274,  0.71590738, -1.        ],
        [-1.        , -1.        , -1.        ,  0.        ,  0.6264872 , 0.65159097,  0.84080142,  0.24749604, -1.        ],
        [-1.        , -1.        , -1.        ,  0.67319672,  0.86455522, 0.        ,  0.        ,  0.        , -1.        ]
    ]))
    assert env.get_reward('prey0') == env.reward_map['prey'][env.ActionStatus.GOOD_HARVEST]
    assert env.get_reward('prey1') == env.reward_map['prey'][env.ActionStatus.BAD_HARVEST]
