import numpy as np
import pytest

from abmarl.examples import MultiCorridor
from abmarl.managers import AllStepManager


def test_init():
    sim = MultiCorridor()
    wrapped_sim = AllStepManager(sim)
    assert wrapped_sim.sim == sim
    assert wrapped_sim.agents == sim.agents


def test_reset_and_step():
    np.random.seed(24)
    sim = AllStepManager(MultiCorridor())

    obs = sim.reset()
    assert sim.sim.corridor[4].id == 'agent3'
    assert sim.sim.corridor[5].id == 'agent4'
    assert sim.sim.corridor[6].id == 'agent2'
    assert sim.sim.corridor[7].id == 'agent1'
    assert sim.sim.corridor[8].id == 'agent0'
    assert sim.done_agents == set()
    assert obs['agent0'] == {'left': [True], 'position': [8], 'right': [False]}
    assert obs['agent1'] == {'left': [True], 'position': [7], 'right': [True]}
    assert obs['agent2'] == {'left': [True], 'position': [6], 'right': [True]}
    assert obs['agent3'] == {'left': [False], 'position': [4], 'right': [True]}
    assert obs['agent4'] == {'left': [True], 'position': [5], 'right': [True]}


    obs, reward, done, _ = sim.step({
        'agent0': MultiCorridor.Actions.RIGHT,
        'agent1': MultiCorridor.Actions.RIGHT,
        'agent2': MultiCorridor.Actions.RIGHT,
        'agent3': MultiCorridor.Actions.RIGHT,
        'agent4': MultiCorridor.Actions.RIGHT,
    })

    assert obs['agent0'] == {'left': [True], 'position': [9], 'right': [False]}
    assert obs['agent1'] == {'left': [True], 'position': [8], 'right': [False]}
    assert obs['agent2'] == {'left': [True], 'position': [7], 'right': [True]}
    assert obs['agent3'] == {'left': [False], 'position': [4], 'right': [False]}
    assert obs['agent4'] == {'left': [False], 'position': [6], 'right': [True]}
    assert reward['agent0'] == 100
    assert reward['agent1'] == -1
    assert reward['agent2'] == -1
    assert reward['agent3'] == -5
    assert reward['agent4'] == -3
    assert done['agent0']
    assert not done['agent1']
    assert not done['agent2']
    assert not done['agent3']
    assert not done['agent4']
    assert not done['__all__']


    with pytest.raises(AssertionError):
        sim.step({
            'agent0': MultiCorridor.Actions.RIGHT,
            'agent1': MultiCorridor.Actions.STAY,
            'agent2': MultiCorridor.Actions.LEFT,
            'agent3': MultiCorridor.Actions.STAY,
            'agent4': MultiCorridor.Actions.LEFT,
        })

    obs, reward, done, _ = sim.step({
        'agent1': MultiCorridor.Actions.STAY,
        'agent2': MultiCorridor.Actions.LEFT,
        'agent3': MultiCorridor.Actions.STAY,
        'agent4': MultiCorridor.Actions.LEFT,
    })

    assert 'agent0' not in obs
    assert obs['agent1'] == {'left': [True], 'position': [8], 'right': [False]}
    assert obs['agent2'] == {'left': [False], 'position': [7], 'right': [True]}
    assert obs['agent3'] == {'left': [False], 'position': [4], 'right': [True]}
    assert obs['agent4'] == {'left': [True], 'position': [5], 'right': [False]}
    assert 'agent0' not in reward
    assert reward['agent1'] == -1
    assert reward['agent2'] == -5
    assert reward['agent3'] == -1
    assert reward['agent4'] == -3
    assert 'agent0' not in done
    assert not done['agent1']
    assert not done['agent2']
    assert not done['agent3']
    assert not done['agent4']
    assert not done['__all__']


    obs, reward, done, _ = sim.step({
        'agent1': MultiCorridor.Actions.RIGHT,
        'agent2': MultiCorridor.Actions.RIGHT,
        'agent3': MultiCorridor.Actions.RIGHT,
        'agent4': MultiCorridor.Actions.LEFT,
    })

    assert obs['agent1'] == {'left': [True], 'position': [9], 'right': [False]}
    assert obs['agent2'] == {'left': [False], 'position': [8], 'right': [False]}
    assert obs['agent3'] == {'left': [False], 'position': [4], 'right': [True]}
    assert obs['agent4'] == {'left': [True], 'position': [5], 'right': [False]}
    assert reward['agent1'] == 100
    assert reward['agent2'] == -1
    assert reward['agent3'] == -7
    assert reward['agent4'] == -7
    assert done['agent1']
    assert not done['agent2']
    assert not done['agent3']
    assert not done['agent4']
    assert not done['__all__']


    with pytest.raises(AssertionError):
        sim.step({
            'agent1': MultiCorridor.Actions.STAY,
            'agent2': MultiCorridor.Actions.STAY,
            'agent3': MultiCorridor.Actions.LEFT,
            'agent4': MultiCorridor.Actions.RIGHT,
        })

    obs, reward, done, _ = sim.step({
        'agent2': MultiCorridor.Actions.STAY,
        'agent3': MultiCorridor.Actions.LEFT,
        'agent4': MultiCorridor.Actions.RIGHT,
    })

    assert 'agent1' not in obs
    assert obs['agent2'] == {'left': [False], 'position': [8], 'right': [False]}
    assert obs['agent3'] == {'left': [False], 'position': [3], 'right': [False]}
    assert obs['agent4'] == {'left': [False], 'position': [6], 'right': [False]}
    assert 'agent1' not in reward
    assert reward['agent2'] == -1
    assert reward['agent3'] == -1
    assert reward['agent4'] == -1
    assert 'agent1' not in done
    assert not done['agent2']
    assert not done['agent3']
    assert not done['agent4']
    assert not done['__all__']


    obs, reward, done, _ = sim.step({
        'agent2': MultiCorridor.Actions.RIGHT,
        'agent3': MultiCorridor.Actions.RIGHT,
        'agent4': MultiCorridor.Actions.RIGHT,
    })

    assert obs['agent2'] == {'left': [False], 'position': [9], 'right': [False]}
    assert obs['agent3'] == {'left': [False], 'position': [4], 'right': [False]}
    assert obs['agent4'] == {'left': [False], 'position': [7], 'right': [False]}
    assert reward['agent2'] == 100
    assert reward['agent3'] == -1
    assert reward['agent4'] == -1
    assert done['agent2']
    assert not done['agent3']
    assert not done['agent4']
    assert not done['__all__']


    with pytest.raises(AssertionError):
        sim.step({
            'agent2': MultiCorridor.Actions.STAY,
            'agent3': MultiCorridor.Actions.RIGHT,
            'agent4': MultiCorridor.Actions.RIGHT,
        })

    obs, reward, done, _ = sim.step({
        'agent3': MultiCorridor.Actions.RIGHT,
        'agent4': MultiCorridor.Actions.RIGHT,
    })

    assert 'agent2' not in obs
    assert obs['agent3'] == {'left': [False], 'position': [5], 'right': [False]}
    assert obs['agent4'] == {'left': [False], 'position': [8], 'right': [False]}
    assert 'agent2' not in reward
    assert reward['agent3'] == -1
    assert reward['agent4'] == -1
    assert 'agent2' not in done
    assert not done['agent3']
    assert not done['agent4']
    assert not done['__all__']


    obs, reward, done, _ = sim.step({
        'agent3': MultiCorridor.Actions.RIGHT,
        'agent4': MultiCorridor.Actions.RIGHT,
    })

    assert obs['agent3'] == {'left': [False], 'position': [6], 'right': [False]}
    assert obs['agent4'] == {'left': [False], 'position': [9], 'right': [False]}
    assert reward['agent3'] == -1
    assert reward['agent4'] == 100
    assert not done['agent3']
    assert done['agent4']
    assert not done['__all__']


    with pytest.raises(AssertionError):
        sim.step({
            'agent3': MultiCorridor.Actions.RIGHT,
            'agent4': MultiCorridor.Actions.STAY,
        })

    obs, reward, done, _ = sim.step({
        'agent3': MultiCorridor.Actions.RIGHT,
    })

    assert 'agent4' not in obs
    assert obs == {'agent3': {'left': [False], 'position': [7], 'right': [False]}}
    assert 'agent4' not in reward
    assert reward == {'agent3': -1,}
    assert 'agent4' not in done
    assert done == {'agent3': False, '__all__': False}


    obs, reward, done, _ = sim.step({
        'agent3': MultiCorridor.Actions.RIGHT,
    })

    assert obs == {'agent3': {'left': [False], 'position': [8], 'right': [False]}}
    assert reward == {'agent3': -1,}
    assert done == {'agent3': False, '__all__': False}


    obs, reward, done, _ = sim.step({
        'agent3': MultiCorridor.Actions.RIGHT,
    })

    assert obs == {'agent3': {'left': [False], 'position': [9], 'right': [False]}}
    assert reward == {'agent3': 100}
    assert done == {'agent3': True, '__all__': True}
