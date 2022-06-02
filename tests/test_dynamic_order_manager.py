
from gym.spaces import Discrete
from matplotlib.pyplot import isinteractive
import pytest

from abmarl.sim import Agent, DynamicOrderSimulation, AgentBasedSimulation
from abmarl.managers import DynamicOrderManager


    class SequentiallyFinishingSim(DynamicOrderSimulation):
        def __init__(self, **kwargs):
            self.agents = {f'agent{i}': PrincipleAgent(id=f'agent{i}') for i in range(4)}

        def reset(self, **kwargs):
            self.next_agent = 'agent3'
            self.done_agents = set()

        def step(self, action_dict, **kwargs):
            next_agents = set()
            for agent_id, action in action_dict.items():
                self.done_agents.add(agent_id)
                for next_ids in action:
                    next_agents.add(next_ids)
            self.next_agent = next_agents

        def render(self, **kwargs):
            pass

        def get_obs(self, agent_id, **kwargs):
            return agent_id

        def get_reward(self, agent_id, **kwargs):
            return {}

        def get_done(self, agent_id, **kwargs):
            return agent_id in self.done_agents

        def get_all_done(self, **kwargs):
            return all([self.get_done(agent_id) for agent_id in self.agents])

        def get_info(self, agent_id, **kwargs):
            return {}


class ABS(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = {
            f'agent{i}': Agent(
                f'agent{i}', observation_space=Discrete(3), action_space=Discrete(4)
            ) for i in range(4)
        }

    def reset(self, **kwargs):
        pass

    def step(self, action_dict, **kwargs):
        pass

    def render(self, **kwargs):
        pass

    def get_obs(self, agent_id, **kwargs):
        return {}

    def get_reward(self, agent_id, **kwargs):
        return {}

    def get_done(self, agent_id, **kwargs):
        return {}

    def get_all_done(self, **kwargs):
        return {}

    def get_info(self, agent_id, **kwargs):
        return {}


def test_dynamic_order_manager_wrong_sim():
    with pytest.raises(AssertionError):
        DynamicOrderManager(ABS())


def test_dynamic_order_manager_reset():
    sim = DynamicOrderManager(
        SequentiallyFinishingSim()
    )
    obs = sim.reset()
    assert obs == {'agent3': 'agent3'}


def test_dynamic_order_manager_stepping():
    sim = DynamicOrderManager(
        SequentiallyFinishingSim()
    )
    sim.reset()
    action = {
        'agent3': [0, 3]
    }
    obs, _, done, _ = sim.step(action)
    assert obs == {'agent0': 'agent0', 'agent3': 'agent3'}
    assert done == {'agent0': False, 'agent3': True, '__all__': False}

    action = {'agent0': [1]}
    obs, _, done, _ = sim.step(action)
    assert obs == {'agent1': 'agent1'}
    assert done == {'agent1': False, '__all__': False}

    action = {'agent1': [0, 2]}
    obs, _, done, _ = sim.step(action)
    assert obs == {'agent0': 'agent0', 'agent2': 'agent2'}
    assert done == {'agent0': True, 'agent2': False, '__all__': False}

    action = {'agent2': [1, 2]}
    obs, _, done, _ = sim.step(action)
    assert obs == {'agent1': 'agent1', 'agent2': 'agent2'}
    assert done == {'agent1': True, 'agent2': True, '__all__': True}


def test_dynamic_order_manager_action_from_already_done_agents():
    sim = DynamicOrderManager(
        SequentiallyFinishingSim()
    )
    sim.reset()
    action = {
        'agent3': [0, 3]
    }
    obs, _, done, _ = sim.step(action)
    assert obs == {'agent0': 'agent0', 'agent3': 'agent3'}
    assert done == {'agent0': False, 'agent3': True, '__all__': False}

    action = {
        'agent3': [3],
        'agent0': [1],
    }
    with pytest.raises(AssertionError):
        sim.step(action)