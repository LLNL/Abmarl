
from gym.spaces import Discrete

from admiral.envs import Agent
from admiral.envs.corridor import Corridor

def test_corridor_attributes():
    assert Corridor.Actions.LEFT == 0
    assert Corridor.Actions.RIGHT == 2
    assert Corridor.Actions.STAY == 1

def test_corridor_build():
    env = Corridor.build()
    assert env.start == 0
    assert env.end == 5
    assert env.agents == {'agent0': Agent('agent0', Discrete(6), Discrete(3))}

def test_corridor_build_end():
    env = Corridor.build({'end': 3})
    assert env.start == 0
    assert env.end == 3
    assert env.agents == {'agent0': Agent('agent0', Discrete(4), Discrete(3))}

def test_corridor_reset():
    env = Corridor.build()
    env.reset()
    assert env.pos == env.start
    obs = env.get_obs('agent0')
    assert obs == 0

def test_corridor_step():
    env = Corridor.build()
    env.reset()

    env.step({'agent0': 0})
    obs = env.get_obs('agent0')
    reward = env.get_reward('agent0')
    done = env.get_done('agent0')
    assert obs == 0
    assert reward == -1
    assert done == False

    env.step({'agent0': 2})
    obs = env.get_obs('agent0')
    reward = env.get_reward('agent0')
    done = env.get_done('agent0')
    assert obs == 1
    assert reward == -1
    assert done == False

    env.step({'agent0': 2})
    obs = env.get_obs('agent0')
    reward = env.get_reward('agent0')
    done = env.get_done('agent0')
    assert obs == 2
    assert reward == -1
    assert done == False

    env.step({'agent0': 1})
    obs = env.get_obs('agent0')
    reward = env.get_reward('agent0')
    done = env.get_done('agent0')
    assert obs == 2
    assert reward == -1
    assert done == False

    env.step({'agent0': 0})
    obs = env.get_obs('agent0')
    reward = env.get_reward('agent0')
    done = env.get_done('agent0')
    assert obs == 1
    assert reward == -1
    assert done == False

    env.step({'agent0': 2})
    obs = env.get_obs('agent0')
    reward = env.get_reward('agent0')
    done = env.get_done('agent0')
    assert obs == 2
    assert reward == -1
    assert done == False

    env.step({'agent0': 2})
    obs = env.get_obs('agent0')
    reward = env.get_reward('agent0')
    done = env.get_done('agent0')
    assert obs == 3
    assert reward == -1
    assert done == False

    env.step({'agent0': 2})
    obs = env.get_obs('agent0')
    reward = env.get_reward('agent0')
    done = env.get_done('agent0')
    assert obs == 4
    assert reward == -1
    assert done == False

    env.step({'agent0': 2})
    obs = env.get_obs('agent0')
    reward = env.get_reward('agent0')
    done = env.get_done('agent0')
    assert obs == 5
    assert reward == 10
    assert done == True
