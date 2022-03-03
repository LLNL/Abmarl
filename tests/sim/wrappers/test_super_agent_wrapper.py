
from gym.spaces import MultiBinary, Discrete, Box, MultiDiscrete, Dict, Tuple
import pytest

from abmarl.sim.agent_based_simulation import AgentBasedSimulation
from abmarl.sim.agent_based_simulation import Agent, PrincipleAgent
from abmarl.sim.wrappers import SuperAgentWrapper
from abmarl.managers import AllStepManager


class SimTest(AgentBasedSimulation):
    def __init__(self):
        self.rewards = [0, 1, 2, 3, 4, 5, 6]
        self.ith_call = -1
        self.dones = [3, 12, 5, 34]
        self.step_count = 0
        self.agents = {
            'agent0': Agent(
                id='agent0',
                observation_space=MultiBinary(4),
                action_space=Tuple((
                    Dict({
                        'first': Discrete(4),
                        'second': Box(low=-1, high=3, shape=(2,), dtype=int)
                    }),
                    MultiBinary(3)
                ))
            ),
            'agent1': Agent(
                id='agent1',
                observation_space=Box(low=0, high=1, shape=(1,), dtype=int),
                action_space=MultiDiscrete([4, 6, 2])
            ),
            'agent2': Agent(
                id='agent2',
                observation_space=MultiDiscrete([2, 2]),
                action_space=Dict({'alpha': MultiBinary(3)})
            ),
            'agent3': Agent(
                id='agent3',
                observation_space=Dict({
                    'first': Discrete(4),
                    'second': Box(low=-1, high=3, shape=(2,), dtype=int)
                }),
                action_space=Tuple((Discrete(3), MultiDiscrete([10, 10]), Discrete(2)))
            ),
            'agent4': PrincipleAgent(
                id='agent4'
            )
        }

    def render(self):
        pass

    def reset(self):
        self.action = {agent.id: None for agent in self.agents.values() if isinstance(agent, Agent)}

    def step(self, action_dict):
        self.step_count += 1
        for agent_id, action in action_dict.items():
            self.action[agent_id] = action

    def get_obs(self, agent_id, **kwargs):
        if agent_id == 'agent0':
            return [0, 0, 0, 1]
        elif agent_id == 'agent1':
            return [0]
        elif agent_id == 'agent2':
            return [1, 0]
        elif agent_id == 'agent3':
            return {'first': 1, 'second': [3, 1]}

    def get_reward(self, agent_id, **kwargs):
        self.ith_call = (self.ith_call + 1) % 7
        return self.rewards[self.ith_call]

    def get_done(self, agent_id, **kwargs):
        return self.step_count >= self.dones[int(agent_id[-1])]

    def get_all_done(self, **kwargs):
        for agent in self.agents.values():
            if not isinstance(agent, Agent): continue
            if not self.get_done(agent.id):
                return False
        return True

    def get_info(self, agent_id, **kwargs):
        return self.action[agent_id]

sim = SuperAgentWrapper(
    SimTest(),
    super_agent_mapping={
        'super0': ['agent0', 'agent3']
    }
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
        SuperAgentWrapper(SimTest(), super_agent_mapping=['agent0'])
    with pytest.raises(AssertionError):
        SuperAgentWrapper(SimTest(), super_agent_mapping={1: ['agent0']})
    with pytest.raises(AssertionError):
        SuperAgentWrapper(SimTest(), super_agent_mapping={'agent0': ['agent1', 'agent2']})
    with pytest.raises(AssertionError):
        SuperAgentWrapper(SimTest(), super_agent_mapping={'super0': 'agent1'})
    with pytest.raises(AssertionError):
        SuperAgentWrapper(SimTest(), super_agent_mapping={'super0': [0, 1]})
    with pytest.raises(AssertionError):
        SuperAgentWrapper(SimTest(), super_agent_mapping={'super0': ['agent5']})
    with pytest.raises(AssertionError):
        SuperAgentWrapper(SimTest(), super_agent_mapping={'super0': ['agent4']})
    with pytest.raises(AssertionError):
        SuperAgentWrapper(
            SimTest(),
            super_agent_mapping={
                'super0': ['agent1', 'agent2'],
                'super1': ['agent0', 'agent1'],
            }
        )


def test_super_agent_mapping_changes_agents():
    tmp_sim = SuperAgentWrapper(
        SimTest(),
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
    assert agents['super0'].action_space['agent0'] == original_agents['agent0'].action_space
    assert agents['super0'].action_space['agent3'] == original_agents['agent3'].action_space
    assert agents['super0'].observation_space['agent0'] == original_agents['agent0'].observation_space
    assert agents['super0'].observation_space['agent3'] == original_agents['agent3'].observation_space

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
    actions = {
        'agent0': ({'first': 2, 'second': [-1, 2]}, [0, 1, 0]),
        'agent3': (0, [7, 3], 1),
        'agent1': [2, 3, 0],
        'agent2': {'alpha': [1, 1, 1]}
    }
    with pytest.raises(AssertionError):
        sim.step(actions)


def test_sim_obs():
    obs = sim.get_obs('super0')
    assert obs in agents['super0'].observation_space
    assert obs == {
        'agent0': [0, 0, 0, 1],
        'agent3': {'first': 1, 'second': [3, 1]}
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
    assert sim.get_reward('super0') == 1
    assert sim.get_reward('agent1') == 2
    assert sim.get_reward('agent2') == 3
    assert sim.get_reward('super0') == 9


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
    assert sim2.agents['double0'].action_space['super0'] == sim2.sim.agents['super0'].action_space
    assert sim2.agents['double0'].action_space['agent1'] == sim2.sim.agents['agent1'].action_space
    assert sim2.agents['double0'].observation_space['super0'] == sim2.sim.agents['super0'].observation_space
    assert sim2.agents['double0'].observation_space['agent1'] == sim2.sim.agents['agent1'].observation_space
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
            'agent3': {'first': 1, 'second': [3, 1]}
        },
        'agent1': [0]
    }
    obs = sim2.get_obs('agent2')
    assert obs in sim2.agents['agent2'].observation_space
    assert obs == [1, 0]

    sim2.unwrapped.ith_call = -1
    assert sim2.get_reward('double0') == 3
    assert sim2.get_reward('agent2') == 3
    assert sim2.get_reward('double0') == 15

    sim2.unwrapped.step_count = 4
    assert not sim2.get_done('double0')
    assert not sim2.get_done('agent2')

    sim2.unwrapped.step_count = 32
    assert not sim2.get_done('double0')
    assert sim2.get_done('agent2')

    sim2.unwrapped.step_count = 40
    assert sim2.get_done('double0')
    assert sim2.get_done('agent2')
