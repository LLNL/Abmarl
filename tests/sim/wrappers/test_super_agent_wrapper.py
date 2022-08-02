
import warnings

from gym.spaces import Discrete, Dict
import pytest

from abmarl.sim.wrappers import SuperAgentWrapper
from abmarl.examples import MultiAgentGymSpacesSim


sim = SuperAgentWrapper(
    MultiAgentGymSpacesSim(),
    super_agent_mapping={
        'super0': ['agent0', 'agent3']
    },
)
agents = sim.agents
original_agents = sim.unwrapped.agents


def test_super_agent_mapping():
    assert sim.super_agent_mapping == {
        'super0': ['agent0', 'agent3']
    }
    assert sim._covered_agents == set(('agent0', 'agent3'))
    assert sim._uncovered_agents == set(('agent1', 'agent2', 'agent4'))


def test_super_agent_mapping_breaks():
    with pytest.raises(AssertionError):
        SuperAgentWrapper(MultiAgentGymSpacesSim(), super_agent_mapping=['agent0'])
    with pytest.raises(AssertionError):
        SuperAgentWrapper(MultiAgentGymSpacesSim(), super_agent_mapping={1: ['agent0']})
    with pytest.raises(AssertionError):
        SuperAgentWrapper(
            MultiAgentGymSpacesSim(), super_agent_mapping={'agent0': ['agent1', 'agent2']}
        )
    with pytest.raises(AssertionError):
        SuperAgentWrapper(MultiAgentGymSpacesSim(), super_agent_mapping={'super0': 'agent1'})
    with pytest.raises(AssertionError):
        SuperAgentWrapper(MultiAgentGymSpacesSim(), super_agent_mapping={'super0': [0, 1]})
    with pytest.raises(AssertionError):
        SuperAgentWrapper(MultiAgentGymSpacesSim(), super_agent_mapping={'super0': ['agent5']})
    with pytest.raises(AssertionError):
        SuperAgentWrapper(MultiAgentGymSpacesSim(), super_agent_mapping={'super0': ['agent4']})
    with pytest.raises(AssertionError):
        SuperAgentWrapper(
            MultiAgentGymSpacesSim(),
            super_agent_mapping={
                'super0': ['agent1', 'agent2'],
                'super1': ['agent0', 'agent1'],
            }
        )


def test_super_agent_mapping_changes_agents():
    tmp_sim = SuperAgentWrapper(
        MultiAgentGymSpacesSim(),
        super_agent_mapping={
            'super0': ['agent0', 'agent3']
        }
    )
    assert tmp_sim.agents.keys() == set(('super0', 'agent1', 'agent2', 'agent4'))
    assert 'agent0' in tmp_sim.agents['super0'].action_space.spaces
    assert 'agent3' in tmp_sim.agents['super0'].action_space.spaces
    tmp_sim.super_agent_mapping = {
        'super0': ['agent1', 'agent0'],
        'super1': ['agent2', 'agent3']
    }
    assert tmp_sim.agents.keys() == set(('super0', 'super1', 'agent4'))
    assert 'agent0' in tmp_sim.agents['super0'].action_space.spaces
    assert 'agent1' in tmp_sim.agents['super0'].action_space.spaces
    assert 'agent2' in tmp_sim.agents['super1'].action_space.spaces
    assert 'agent3' in tmp_sim.agents['super1'].action_space.spaces


def test_agent_spaces():
    assert len(agents) == 4
    assert agents['super0'].action_space == Dict({
        'agent0': original_agents['agent0'].action_space,
        'agent3': original_agents['agent3'].action_space,
    })
    assert agents['super0'].observation_space == Dict({
        'agent0': original_agents['agent0'].observation_space,
        'agent3': original_agents['agent3'].observation_space,
        'mask': Dict({'agent0': Discrete(2), 'agent3': Discrete(2)})
    })
    assert agents['agent1'] == original_agents['agent1']
    assert agents['agent2'] == original_agents['agent2']
    assert agents['agent4'] == original_agents['agent4']


def test_sim_step():
    sim.reset()
    assert sim.unwrapped.action == {
        'agent0': None,
        'agent1': None,
        'agent2': None,
        'agent3': None
    }

    actions = {
        'super0': {
            'agent0': ({'first': 2, 'second': [-1, 2]}, [0, 1, 0]),
            'agent3': (0, [7, 3], 1)
        },
        'agent1': [2, 3, 0],
        'agent2': {'alpha': [1, 1, 1]}
    }
    sim.step(actions)
    assert sim.unwrapped.action == {
        'agent0': ({'first': 2, 'second': [-1, 2]}, [0, 1, 0]),
        'agent3': (0, [7, 3], 1),
        'agent1': [2, 3, 0],
        'agent2': {'alpha': [1, 1, 1]}
    }


def test_sim_step_breaks():
    sim.reset()
    actions = {
        'agent0': ({'first': 2, 'second': [-1, 2]}, [0, 1, 0]),
        'agent3': (0, [7, 3], 1),
        'agent1': [2, 3, 0],
        'agent2': {'alpha': [1, 1, 1]}
    }
    with pytest.raises(AssertionError):
        sim.step(actions)


def test_sim_step_covered_agent_done():
    sim.reset()
    sim.unwrapped.step_count = 4
    actions = {
        'super0': {
            'agent0': ({'first': 2, 'second': [-1, 2]}, [0, 1, 0]),
            'agent3': (0, [7, 3], 1)
        },
        'agent1': [2, 3, 0],
        'agent2': {'alpha': [1, 1, 1]}
    }
    sim.step(actions)
    assert sim.unwrapped.action == {
        'agent0': None,
        'agent3': (0, [7, 3], 1),
        'agent1': [2, 3, 0],
        'agent2': {'alpha': [1, 1, 1]}
    }


def test_sim_obs():
    sim.unwrapped.step_count = 4
    obs = sim.get_obs('super0')
    assert obs in agents['super0'].observation_space
    assert obs == {
        'agent0': [0, 0, 0, 1],
        'agent3': {'first': 1, 'second': [3, 1]},
        'mask': {'agent0': False, 'agent3': True}
    }

    obs = sim.get_obs('agent1')
    assert obs in agents['agent1'].observation_space
    assert obs == [0]

    obs = sim.get_obs('agent2')
    assert obs in agents['agent2'].observation_space
    assert obs == [1, 0]


def test_sim_obs_breaks():
    with pytest.raises(AssertionError):
        sim.get_obs('agent0')
    with pytest.raises(AssertionError):
        sim.get_obs('agent3')


def test_sim_rewards():
    sim.unwrapped.step_count = 0
    assert sim.get_reward('super0') == 9
    assert sim.get_reward('agent1') == 3
    assert sim.get_reward('agent2') == 5


def test_sim_rewards_breaks():
    with pytest.raises(AssertionError):
        sim.get_reward('agent0')
    with pytest.raises(AssertionError):
        sim.get_reward('agent3')


def test_sim_done():
    sim.unwrapped.step_count = 10
    assert not sim.get_done('super0')
    assert not sim.get_done('agent1')
    assert sim.get_done('agent2')

    sim.unwrapped.step_count = 40
    assert sim.get_done('super0')
    assert sim.get_done('agent1')


def test_sim_done_breaks():
    with pytest.raises(AssertionError):
        sim.get_done('agent0')
    with pytest.raises(AssertionError):
        sim.get_done('agent3')


def test_sim_all_done():
    sim.unwrapped.step_count = 15
    assert not sim.get_all_done()
    sim.unwrapped.step_count = 40
    assert sim.get_all_done()


def test_double_wrap():
    sim2 = SuperAgentWrapper(
        sim,
        super_agent_mapping={
            'double0': ['super0', 'agent1']
        }
    )
    assert sim2._covered_agents == set(('super0', 'agent1'))
    assert sim2._uncovered_agents == set(('agent2', 'agent4'))
    assert sim2.agents.keys() == set(('double0', 'agent2', 'agent4'))
    assert 'super0' in sim2.agents['double0'].action_space.spaces
    assert 'agent1' in sim2.agents['double0'].action_space.spaces

    assert len(sim2.agents) == 3
    assert sim2.agents['double0'].action_space == Dict({
        'super0': sim2.sim.agents['super0'].action_space,
        'agent1': sim2.sim.agents['agent1'].action_space
    })
    assert sim2.agents['double0'].observation_space == Dict({
        'super0': sim2.sim.agents['super0'].observation_space,
        'agent1': sim2.sim.agents['agent1'].observation_space,
        'mask': Dict({'super0': Discrete(2), 'agent1': Discrete(2)})
    })
    assert sim2.agents['agent2'] == original_agents['agent2']
    assert sim2.agents['agent4'] == original_agents['agent4']

    sim2.reset()
    assert sim2.unwrapped.action == {
        'agent0': None,
        'agent1': None,
        'agent2': None,
        'agent3': None
    }
    actions = {
        'double0': {
            'super0': {
                'agent0': ({'first': 2, 'second': [-1, 2]}, [0, 1, 0]),
                'agent3': (0, [7, 3], 1)
            },
            'agent1': [2, 3, 0],
        },
        'agent2': {'alpha': [1, 1, 1]}
    }
    sim2.step(actions)
    assert sim2.unwrapped.action == {
        'agent0': ({'first': 2, 'second': [-1, 2]}, [0, 1, 0]),
        'agent3': (0, [7, 3], 1),
        'agent1': [2, 3, 0],
        'agent2': {'alpha': [1, 1, 1]}
    }

    obs = sim2.get_obs('double0')
    assert obs in sim2.agents['double0'].observation_space
    assert obs == {
        'super0': {
            'agent0': [0, 0, 0, 1],
            'agent3': {'first': 1, 'second': [3, 1]},
            'mask': {'agent0': True, 'agent3': True}
        },
        'agent1': [0],
        'mask': {'agent1': True, 'super0': True}
    }
    obs = sim2.get_obs('agent2')
    assert obs in sim2.agents['agent2'].observation_space
    assert obs == [1, 0]

    assert sim2.get_reward('double0') == 12
    assert sim2.get_reward('agent2') == 5

    sim2.unwrapped.step_count = 4
    assert not sim2.get_done('double0')
    assert not sim2.get_done('agent2')

    sim2.unwrapped.step_count = 32
    assert not sim2.get_done('double0')
    assert sim2.get_done('agent2')

    sim2.unwrapped.step_count = 40
    assert sim2.get_done('double0')
    assert sim2.get_done('agent2')


def test_using_null_obs_when_done():
    sim.reset()
    sim.unwrapped.step_count = 2
    obs = sim.get_obs('super0')
    assert obs in agents['super0'].observation_space
    assert obs == {
        'agent0': [0, 0, 0, 1],
        'agent3': {'first': 1, 'second': [3, 1]},
        'mask': {'agent0': True, 'agent3': True}
    }


    sim.step({'agent1': [2, 2, 0]})
    assert sim.unwrapped.step_count == 3
    assert sim.unwrapped.get_done('agent0')
    assert not sim.unwrapped.get_done('agent3')

    sim.step({'agent1': [2, 2, 0]})
    assert sim.unwrapped.step_count == 4
    assert sim.unwrapped.get_done('agent0')
    assert not sim.unwrapped.get_done('agent3')

    assert sim.get_obs('super0') == {
        'agent0': [0, 0, 0, 1],
        'agent3': {'first': 1, 'second': [3, 1]},
        'mask': {'agent0': False, 'agent3': True}
    }
    assert sim.get_obs('super0') == {
        'agent0': [0, 0, 0, 0],
        'agent3': {'first': 1, 'second': [3, 1]},
        'mask': {'agent0': False, 'agent3': True}
    }
    assert sim.unwrapped.get_obs('agent0') == [0, 0, 0, 1]

    assert sim.get_reward('super0') == 9
    assert sim.get_reward('super0') == 7
    assert sim.unwrapped.get_reward('agent0') == 2


    sim.step({'agent2': {'alpha': [1, 1, 0]}})
    sim.unwrapped.step_count = 35
    assert sim.unwrapped.get_done('agent0')
    assert sim.unwrapped.get_done('agent3')

    assert sim.get_obs('super0') == {
        'agent0': [0, 0, 0, 0],
        'agent3': {'first': 1, 'second': [3, 1]},
        'mask': {'agent0': False, 'agent3': False}
    }
    assert sim.get_obs('super0') == {
        'agent0': [0, 0, 0, 0],
        'agent3': {'first': 1, 'second': [3, 1]},
        'mask': {'agent0': False, 'agent3': False}
    }
    assert sim.get_reward('super0') == 7
    assert sim.get_reward('super0') == 0
    assert sim.unwrapped.get_reward('agent3') == 7


def test_null_obs_warning():
    sim.reset()
    sim._warning_issued = False
    sim.unwrapped.step_count = 35
    assert sim.unwrapped.get_done('agent3')
    sim.get_obs('super0') # Get the last observations

    # Now get the null observations
    with pytest.warns(
        UserWarning,
        match=r"Some covered agents in the SuperAgentWrapper do not specify a null observation."
    ):
        sim.get_obs('super0')

    # Ensure warning is only given once
    with warnings.catch_warnings():
        sim.get_obs('super0')
        warnings.simplefilter("error")
