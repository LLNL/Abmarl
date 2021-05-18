
from gym.spaces import Box, MultiBinary, Discrete, Dict
import numpy as np

from admiral.envs import Agent
from admiral.envs.corridor import MultiCorridor as Corridor


def test_corridor_attributes():
    assert Corridor.Actions.LEFT == 0
    assert Corridor.Actions.RIGHT == 2
    assert Corridor.Actions.STAY == 1

def test_corridor_build():
    env = Corridor.build()
    assert env.end == 10
    assert env.agents == {
        'agent0': Agent(id='agent0', observation_space=Dict({'position': Box(0,9,(1,),np.int), 'left': MultiBinary(1), 'right': MultiBinary(1)}), action_space=Discrete(3)),
        'agent1': Agent(id='agent1', observation_space=Dict({'position': Box(0,9,(1,),np.int), 'left': MultiBinary(1), 'right': MultiBinary(1)}), action_space=Discrete(3)),
        'agent2': Agent(id='agent2', observation_space=Dict({'position': Box(0,9,(1,),np.int), 'left': MultiBinary(1), 'right': MultiBinary(1)}), action_space=Discrete(3)),
        'agent3': Agent(id='agent3', observation_space=Dict({'position': Box(0,9,(1,),np.int), 'left': MultiBinary(1), 'right': MultiBinary(1)}), action_space=Discrete(3)),
        'agent4': Agent(id='agent4', observation_space=Dict({'position': Box(0,9,(1,),np.int), 'left': MultiBinary(1), 'right': MultiBinary(1)}), action_space=Discrete(3)),
    }

def test_corridor_build_end():
    env = Corridor.build({'end': 7})
    assert env.end == 7
    for agent in env.agents.values():
        assert agent.observation_space['position'] == Box(0, 6, (1,), np.int)

def test_corridor_build_num_agents():
    env = Corridor.build({'num_agents': 2})
    assert env.agents == {
        'agent0': Agent(id='agent0', observation_space=Dict({'position': Box(0,9,(1,),np.int), 'left': MultiBinary(1), 'right': MultiBinary(1)}), action_space=Discrete(3)),
        'agent1': Agent(id='agent1', observation_space=Dict({'position': Box(0,9,(1,),np.int), 'left': MultiBinary(1), 'right': MultiBinary(1)}), action_space=Discrete(3)),
    }

def test_corridor_reset():
    np.random.seed(24)
    env = Corridor.build()
    env.reset()
    for agent in env.agents.values():
        assert agent == env.corridor[agent.position]
    assert env.reward == {agent_id: 0 for agent_id in env.agents}

def test_corridor_step():
    np.random.seed(24)
    env = Corridor.build()
    env.reset()
    assert env.corridor[4].id == 'agent3'
    assert env.corridor[5].id == 'agent4'
    assert env.corridor[6].id == 'agent2'
    assert env.corridor[7].id == 'agent1'
    assert env.corridor[8].id == 'agent0'

    # Get observation and make some assertion
    assert env.get_obs('agent0') == {'left': [True], 'position': 8, 'right': [False]}
    assert env.get_obs('agent1') == {'left': [True], 'position': 7, 'right': [True]}
    assert env.get_obs('agent2') == {'left': [True], 'position': 6, 'right': [True]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [True]}
    assert env.get_obs('agent4') == {'left': [True], 'position': 5, 'right': [True]}

    env.step({'agent0': Corridor.Actions.RIGHT})
    assert env.corridor[8] == None
    assert env.corridor[9] == None
    assert env.get_obs('agent0') == {'left': [False], 'position': 9, 'right': [False]}
    assert env.get_obs('agent1') == {'left': [True], 'position': 7, 'right': [False]}
    assert env.get_obs('agent2') == {'left': [True], 'position': 6, 'right': [True]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [True]}
    assert env.get_obs('agent4') == {'left': [True], 'position': 5, 'right': [True]}
    assert env.get_reward('agent0') == 100
    assert env.get_done('agent0') == True

    env.step({'agent1': Corridor.Actions.RIGHT})
    assert env.corridor[7] == None
    assert env.corridor[8].id == 'agent1'
    assert env.get_obs('agent1') == {'left': [False], 'position': 8, 'right': [False]}
    assert env.get_obs('agent2') == {'left': [True], 'position': 6, 'right': [False]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [True]}
    assert env.get_obs('agent4') == {'left': [True], 'position': 5, 'right': [True]}
    assert env.get_reward('agent1') == -1
    assert env.get_done('agent1') == False

    env.step({'agent2': Corridor.Actions.RIGHT})
    assert env.corridor[6] == None
    assert env.corridor[7].id == 'agent2'
    assert env.get_obs('agent1') == {'left': [True], 'position': 8, 'right': [False]}
    assert env.get_obs('agent2') == {'left': [False], 'position': 7, 'right': [True]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [True]}
    assert env.get_obs('agent4') == {'left': [True], 'position': 5, 'right': [False]}
    assert env.get_reward('agent2') == -1
    assert env.get_done('agent2') == False

    env.step({'agent3': Corridor.Actions.RIGHT})
    assert env.corridor[4].id == 'agent3'
    assert env.corridor[5].id == 'agent4'
    assert env.get_obs('agent1') == {'left': [True], 'position': 8, 'right': [False]}
    assert env.get_obs('agent2') == {'left': [False], 'position': 7, 'right': [True]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [True]}
    assert env.get_obs('agent4') == {'left': [True], 'position': 5, 'right': [False]}
    assert env.get_reward('agent3') == -5
    assert env.get_reward('agent4') == -2
    assert env.get_done('agent3') == False

    env.step({'agent4': Corridor.Actions.RIGHT})
    assert env.corridor[5] == None
    assert env.corridor[6].id == 'agent4'
    assert env.get_obs('agent1') == {'left': [True], 'position': 8, 'right': [False]}
    assert env.get_obs('agent2') == {'left': [True], 'position': 7, 'right': [True]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [False]}
    assert env.get_obs('agent4') == {'left': [False], 'position': 6, 'right': [True]}
    assert env.get_reward('agent4') == -1
    assert env.get_done('agent4') == False

    env.step({'agent1': Corridor.Actions.STAY})
    assert env.corridor[8].id == 'agent1'
    assert env.get_obs('agent1') == {'left': [True], 'position': 8, 'right': [False]}
    assert env.get_obs('agent2') == {'left': [True], 'position': 7, 'right': [True]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [False]}
    assert env.get_obs('agent4') == {'left': [False], 'position': 6, 'right': [True]}
    assert env.get_reward('agent1') == -1
    assert env.get_done('agent1') == False

    env.step({'agent2': Corridor.Actions.LEFT})
    assert env.corridor[7].id == 'agent2'
    assert env.corridor[6].id == 'agent4'
    assert env.get_obs('agent1') == {'left': [True], 'position': 8, 'right': [False]}
    assert env.get_obs('agent2') == {'left': [True], 'position': 7, 'right': [True]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [False]}
    assert env.get_obs('agent4') == {'left': [False], 'position': 6, 'right': [True]}
    assert env.get_reward('agent2') == -5
    assert env.get_done('agent2') == False

    env.step({'agent3': Corridor.Actions.STAY})
    assert env.corridor[4].id == 'agent3'
    assert env.get_obs('agent1') == {'left': [True], 'position': 8, 'right': [False]}
    assert env.get_obs('agent2') == {'left': [True], 'position': 7, 'right': [True]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [False]}
    assert env.get_obs('agent4') == {'left': [False], 'position': 6, 'right': [True]}
    assert env.get_reward('agent3') == -1
    assert env.get_done('agent3') == False

    env.step({'agent4': Corridor.Actions.LEFT})
    assert env.corridor[5].id == 'agent4'
    assert env.get_obs('agent1') == {'left': [True], 'position': 8, 'right': [False]}
    assert env.get_obs('agent2') == {'left': [False], 'position': 7, 'right': [True]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [True]}
    assert env.get_obs('agent4') == {'left': [True], 'position': 5, 'right': [False]}
    assert env.get_reward('agent4') == -1
    assert env.get_done('agent4') == False

    env.step({'agent1': Corridor.Actions.RIGHT})
    assert env.corridor[9] == None
    assert env.get_obs('agent1') == {'left': [False], 'position': 9, 'right': [False]}
    assert env.get_obs('agent2') == {'left': [False], 'position': 7, 'right': [False]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [True]}
    assert env.get_obs('agent4') == {'left': [True], 'position': 5, 'right': [False]}
    assert env.get_reward('agent1') == 100
    assert env.get_done('agent1') == True

    env.step({'agent2': Corridor.Actions.RIGHT})
    assert env.corridor[8].id == 'agent2'
    assert env.corridor[7] == None
    assert env.get_obs('agent2') == {'left': [False], 'position': 8, 'right': [False]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [True]}
    assert env.get_obs('agent4') == {'left': [True], 'position': 5, 'right': [False]}
    assert env.get_reward('agent2') == -1
    assert env.get_done('agent2') == False

    env.step({'agent3': Corridor.Actions.RIGHT})
    assert env.corridor[4].id == 'agent3'
    assert env.corridor[5].id == 'agent4'
    assert env.get_obs('agent2') == {'left': [False], 'position': 8, 'right': [False]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [True]}
    assert env.get_obs('agent4') == {'left': [True], 'position': 5, 'right': [False]}
    assert env.get_reward('agent3') == -5
    assert env.get_reward('agent4') == -3
    assert env.get_done('agent3') == False

    env.step({'agent4': Corridor.Actions.LEFT})
    assert env.corridor[4].id == 'agent3'
    assert env.corridor[5].id == 'agent4'
    assert env.get_obs('agent2') == {'left': [False], 'position': 8, 'right': [False]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [True]}
    assert env.get_obs('agent4') == {'left': [True], 'position': 5, 'right': [False]}
    assert env.get_reward('agent3') == -7
    assert env.get_reward('agent4') == -5
    assert env.get_done('agent4') == False

    env.step({'agent2': Corridor.Actions.STAY})
    assert env.corridor[8].id == 'agent2'
    assert env.get_obs('agent2') == {'left': [False], 'position': 8, 'right': [False]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 4, 'right': [True]}
    assert env.get_obs('agent4') == {'left': [True], 'position': 5, 'right': [False]}
    assert env.get_reward('agent2') == -1
    assert env.get_done('agent2') == False

    env.step({'agent3': Corridor.Actions.LEFT})
    assert env.corridor[4] == None
    assert env.corridor[3].id == 'agent3'
    assert env.get_obs('agent2') == {'left': [False], 'position': 8, 'right': [False]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 3, 'right': [False]}
    assert env.get_obs('agent4') == {'left': [False], 'position': 5, 'right': [False]}
    assert env.get_reward('agent3') == -1
    assert env.get_done('agent3') == False

    env.step({'agent4': Corridor.Actions.RIGHT})
    assert env.corridor[5] == None
    assert env.corridor[6].id == 'agent4'
    assert env.get_obs('agent2') == {'left': [False], 'position': 8, 'right': [False]}
    assert env.get_obs('agent3') == {'left': [False], 'position': 3, 'right': [False]}
    assert env.get_obs('agent4') == {'left': [False], 'position': 6, 'right': [False]}
    assert env.get_reward('agent4') == -1
    assert env.get_done('agent4') == False

    assert env.get_all_done() == False
    env.step({'agent2': Corridor.Actions.RIGHT})
    env.step({'agent4': Corridor.Actions.RIGHT})
    env.step({'agent4': Corridor.Actions.RIGHT})
    env.step({'agent4': Corridor.Actions.RIGHT})
    env.step({'agent3': Corridor.Actions.RIGHT})
    env.step({'agent3': Corridor.Actions.RIGHT})
    env.step({'agent3': Corridor.Actions.RIGHT})
    env.step({'agent3': Corridor.Actions.RIGHT})
    env.step({'agent3': Corridor.Actions.RIGHT})
    env.step({'agent3': Corridor.Actions.RIGHT})
    assert env.get_all_done() == True
