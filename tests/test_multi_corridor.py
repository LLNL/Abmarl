from gym.spaces import Box, MultiBinary, Discrete, Dict
import numpy as np

from abmarl.sim import Agent
from abmarl.examples import MultiCorridor


def test_corridor_attributes():
    assert MultiCorridor.Actions.LEFT == 0
    assert MultiCorridor.Actions.RIGHT == 2
    assert MultiCorridor.Actions.STAY == 1


def test_corridor_init():
    sim = MultiCorridor()
    assert sim.end == 10
    assert sim.agents == {
        'agent0': Agent(
            id='agent0',
            observation_space=Dict({
                'position': Box(0,9,(1,),int),
                'left': MultiBinary(1),
                'right': MultiBinary(1)
            }), action_space=Discrete(3)),
        'agent1': Agent(
            id='agent1',
            observation_space=Dict({
                'position': Box(0,9,(1,),int),
                'left': MultiBinary(1),
                'right': MultiBinary(1)
            }), action_space=Discrete(3)),
        'agent2': Agent(
            id='agent2',
            observation_space=Dict({
                'position': Box(0,9,(1,),int),
                'left': MultiBinary(1),
                'right': MultiBinary(1)
            }), action_space=Discrete(3)),
        'agent3': Agent(
            id='agent3',
            observation_space=Dict({
                'position': Box(0,9,(1,),int),
                'left': MultiBinary(1),
                'right': MultiBinary(1)
            }), action_space=Discrete(3)),
        'agent4': Agent(
            id='agent4',
            observation_space=Dict({
                'position': Box(0,9,(1,),int),
                'left': MultiBinary(1),
                'right': MultiBinary(1)
            }), action_space=Discrete(3)),
    }


def test_corridor_init_end():
    sim = MultiCorridor(end=7)
    assert sim.end == 7
    for agent in sim.agents.values():
        assert agent.observation_space['position'] == Box(0, 6, (1,), int)


def test_corridor_init_num_agents():
    sim = MultiCorridor(num_agents=2)
    assert sim.agents == {
        'agent0': Agent(
            id='agent0',
            observation_space=Dict({
                'position': Box(0,9,(1,),int), 'left': MultiBinary(1), 'right': MultiBinary(1)
            }), action_space=Discrete(3)),
        'agent1': Agent(
            id='agent1',
            observation_space=Dict({
                'position': Box(0,9,(1,),int), 'left': MultiBinary(1), 'right': MultiBinary(1)
            }), action_space=Discrete(3)),
    }


def test_corridor_reset():
    np.random.seed(24)
    sim = MultiCorridor()
    sim.reset()
    for agent in sim.agents.values():
        assert agent == sim.corridor[agent.position]
    assert sim.reward == {agent_id: 0 for agent_id in sim.agents}


def test_corridor_step():
    np.random.seed(24)
    sim = MultiCorridor()
    sim.reset()
    assert sim.corridor[4].id == 'agent3'
    assert sim.corridor[5].id == 'agent4'
    assert sim.corridor[6].id == 'agent2'
    assert sim.corridor[7].id == 'agent1'
    assert sim.corridor[8].id == 'agent0'

    # Get observation and make some assertion
    assert sim.get_obs('agent0') == {'left': [True], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent1') == {'left': [True], 'position': [7], 'right': [True]}
    assert sim.get_obs('agent2') == {'left': [True], 'position': [6], 'right': [True]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [True]}
    assert sim.get_obs('agent4') == {'left': [True], 'position': [5], 'right': [True]}

    sim.step({'agent0': MultiCorridor.Actions.RIGHT})
    assert sim.corridor[8] is None
    assert sim.corridor[9] is None
    assert sim.get_obs('agent0') == {'left': [False], 'position': [9], 'right': [False]}
    assert sim.get_obs('agent1') == {'left': [True], 'position': [7], 'right': [False]}
    assert sim.get_obs('agent2') == {'left': [True], 'position': [6], 'right': [True]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [True]}
    assert sim.get_obs('agent4') == {'left': [True], 'position': [5], 'right': [True]}
    assert sim.get_reward('agent0') == 100
    assert sim.get_done('agent0')

    sim.step({'agent1': MultiCorridor.Actions.RIGHT})
    assert sim.corridor[7] is None
    assert sim.corridor[8].id == 'agent1'
    assert sim.get_obs('agent1') == {'left': [False], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent2') == {'left': [True], 'position': [6], 'right': [False]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [True]}
    assert sim.get_obs('agent4') == {'left': [True], 'position': [5], 'right': [True]}
    assert sim.get_reward('agent1') == -1
    assert not sim.get_done('agent1')

    sim.step({'agent2': MultiCorridor.Actions.RIGHT})
    assert sim.corridor[6] is None
    assert sim.corridor[7].id == 'agent2'
    assert sim.get_obs('agent1') == {'left': [True], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent2') == {'left': [False], 'position': [7], 'right': [True]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [True]}
    assert sim.get_obs('agent4') == {'left': [True], 'position': [5], 'right': [False]}
    assert sim.get_reward('agent2') == -1
    assert not sim.get_done('agent2')

    sim.step({'agent3': MultiCorridor.Actions.RIGHT})
    assert sim.corridor[4].id == 'agent3'
    assert sim.corridor[5].id == 'agent4'
    assert sim.get_obs('agent1') == {'left': [True], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent2') == {'left': [False], 'position': [7], 'right': [True]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [True]}
    assert sim.get_obs('agent4') == {'left': [True], 'position': [5], 'right': [False]}
    assert sim.get_reward('agent3') == -5
    assert not sim.get_done('agent3')

    sim.step({'agent4': MultiCorridor.Actions.RIGHT})
    assert sim.corridor[5] is None
    assert sim.corridor[6].id == 'agent4'
    assert sim.get_obs('agent1') == {'left': [True], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent2') == {'left': [True], 'position': [7], 'right': [True]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [False]}
    assert sim.get_obs('agent4') == {'left': [False], 'position': [6], 'right': [True]}
    assert sim.get_reward('agent4') == -3
    assert not sim.get_done('agent4')

    sim.step({'agent1': MultiCorridor.Actions.STAY})
    assert sim.corridor[8].id == 'agent1'
    assert sim.get_obs('agent1') == {'left': [True], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent2') == {'left': [True], 'position': [7], 'right': [True]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [False]}
    assert sim.get_obs('agent4') == {'left': [False], 'position': [6], 'right': [True]}
    assert sim.get_reward('agent1') == -1
    assert not sim.get_done('agent1')

    sim.step({'agent2': MultiCorridor.Actions.LEFT})
    assert sim.corridor[7].id == 'agent2'
    assert sim.corridor[6].id == 'agent4'
    assert sim.get_obs('agent1') == {'left': [True], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent2') == {'left': [True], 'position': [7], 'right': [True]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [False]}
    assert sim.get_obs('agent4') == {'left': [False], 'position': [6], 'right': [True]}
    assert sim.get_reward('agent2') == -5
    assert not sim.get_done('agent2')

    sim.step({'agent3': MultiCorridor.Actions.STAY})
    assert sim.corridor[4].id == 'agent3'
    assert sim.get_obs('agent1') == {'left': [True], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent2') == {'left': [True], 'position': [7], 'right': [True]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [False]}
    assert sim.get_obs('agent4') == {'left': [False], 'position': [6], 'right': [True]}
    assert sim.get_reward('agent3') == -1
    assert not sim.get_done('agent3')

    sim.step({'agent4': MultiCorridor.Actions.LEFT})
    assert sim.corridor[5].id == 'agent4'
    assert sim.get_obs('agent1') == {'left': [True], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent2') == {'left': [False], 'position': [7], 'right': [True]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [True]}
    assert sim.get_obs('agent4') == {'left': [True], 'position': [5], 'right': [False]}
    assert sim.get_reward('agent4') == -3
    assert not sim.get_done('agent4')

    sim.step({'agent1': MultiCorridor.Actions.RIGHT})
    assert sim.corridor[9] is None
    assert sim.get_obs('agent1') == {'left': [False], 'position': [9], 'right': [False]}
    assert sim.get_obs('agent2') == {'left': [False], 'position': [7], 'right': [False]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [True]}
    assert sim.get_obs('agent4') == {'left': [True], 'position': [5], 'right': [False]}
    assert sim.get_reward('agent1') == 100
    assert sim.get_done('agent1')

    sim.step({'agent2': MultiCorridor.Actions.RIGHT})
    assert sim.corridor[8].id == 'agent2'
    assert sim.corridor[7] is None
    assert sim.get_obs('agent2') == {'left': [False], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [True]}
    assert sim.get_obs('agent4') == {'left': [True], 'position': [5], 'right': [False]}
    assert sim.get_reward('agent2') == -1
    assert not sim.get_done('agent2')

    sim.step({'agent3': MultiCorridor.Actions.RIGHT})
    assert sim.corridor[4].id == 'agent3'
    assert sim.corridor[5].id == 'agent4'
    assert sim.get_obs('agent2') == {'left': [False], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [True]}
    assert sim.get_obs('agent4') == {'left': [True], 'position': [5], 'right': [False]}
    assert sim.get_reward('agent3') == -5
    assert not sim.get_done('agent3')

    sim.step({'agent4': MultiCorridor.Actions.LEFT})
    assert sim.corridor[4].id == 'agent3'
    assert sim.corridor[5].id == 'agent4'
    assert sim.get_obs('agent2') == {'left': [False], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [True]}
    assert sim.get_obs('agent4') == {'left': [True], 'position': [5], 'right': [False]}
    assert sim.get_reward('agent4') == -7
    assert not sim.get_done('agent4')

    sim.step({'agent2': MultiCorridor.Actions.STAY})
    assert sim.corridor[8].id == 'agent2'
    assert sim.get_obs('agent2') == {'left': [False], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [4], 'right': [True]}
    assert sim.get_obs('agent4') == {'left': [True], 'position': [5], 'right': [False]}
    assert sim.get_reward('agent2') == -1
    assert not sim.get_done('agent2')

    sim.step({'agent3': MultiCorridor.Actions.LEFT})
    assert sim.corridor[4] is None
    assert sim.corridor[3].id == 'agent3'
    assert sim.get_obs('agent2') == {'left': [False], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [3], 'right': [False]}
    assert sim.get_obs('agent4') == {'left': [False], 'position': [5], 'right': [False]}
    assert sim.get_reward('agent3') == -3
    assert not sim.get_done('agent3')

    sim.step({'agent4': MultiCorridor.Actions.RIGHT})
    assert sim.corridor[5] is None
    assert sim.corridor[6].id == 'agent4'
    assert sim.get_obs('agent2') == {'left': [False], 'position': [8], 'right': [False]}
    assert sim.get_obs('agent3') == {'left': [False], 'position': [3], 'right': [False]}
    assert sim.get_obs('agent4') == {'left': [False], 'position': [6], 'right': [False]}
    assert sim.get_reward('agent4') == -1
    assert not sim.get_done('agent4')

    assert not sim.get_all_done()
    sim.step({'agent2': MultiCorridor.Actions.RIGHT})
    sim.step({'agent4': MultiCorridor.Actions.RIGHT})
    sim.step({'agent4': MultiCorridor.Actions.RIGHT})
    sim.step({'agent4': MultiCorridor.Actions.RIGHT})
    sim.step({'agent3': MultiCorridor.Actions.RIGHT})
    sim.step({'agent3': MultiCorridor.Actions.RIGHT})
    sim.step({'agent3': MultiCorridor.Actions.RIGHT})
    sim.step({'agent3': MultiCorridor.Actions.RIGHT})
    sim.step({'agent3': MultiCorridor.Actions.RIGHT})
    sim.step({'agent3': MultiCorridor.Actions.RIGHT})
    assert sim.get_all_done()
