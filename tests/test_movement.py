
import numpy as np
import pytest

from admiral.component_envs.world import GridWorldEnv, WorldAgent
from admiral.component_envs.movement import GridMovementEnv, GridMovementAgent

class GridWorldMovementAgent(WorldAgent, GridMovementAgent):
    pass

class GridWorldMovementEnv(GridWorldEnv, GridMovementEnv):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.world = GridWorldEnv(**kwargs)
        self.movement = GridMovementEnv(**kwargs)
    
    def reset(self, **kwargs):
        self.world.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if 'move' in action:
                agent.position = self.movement.process_move(agent.position, action['move'])
        
def test_agents_moving_in_grid():
    agents = {
        'agent0': GridWorldMovementAgent(id='agent0', starting_position=np.array([6, 4]), move = 2),
        'agent1': GridWorldMovementAgent(id='agent1', starting_position=np.array([3, 3]), move = 3),
        'agent2': GridWorldMovementAgent(id='agent2', starting_position=np.array([0, 1]), move = 1),
        'agent3': GridWorldMovementAgent(id='agent3', starting_position=np.array([8, 4]), move = 1),
    }
    env = GridWorldMovementEnv(region=10, agents=agents)
    env.reset()

    env.step({
        'agent0': {'move': np.array([-1,  1])},
        'agent1': {'move': np.array([ 0,  1])},
        'agent2': {'move': np.array([ 0, -1])},
        'agent3': {'move': np.array([ 1,  0])},
    })
    np.testing.assert_array_equal(env.agents['agent0'].position, np.array([5, 5]))
    np.testing.assert_array_equal(env.agents['agent1'].position, np.array([3, 4]))
    np.testing.assert_array_equal(env.agents['agent2'].position, np.array([0, 0]))
    np.testing.assert_array_equal(env.agents['agent3'].position, np.array([9, 4]))

    env.step({
        'agent0': {'move': np.array([ 2, -2])},
        'agent1': {'move': np.array([-3,  0])},
        'agent2': {'move': np.array([-1,  0])},
        'agent3': {'move': np.array([ 1,  1])},
    })
    np.testing.assert_array_equal(env.agents['agent0'].position, np.array([7, 3]))
    np.testing.assert_array_equal(env.agents['agent1'].position, np.array([0, 4]))
    np.testing.assert_array_equal(env.agents['agent2'].position, np.array([0, 0]))
    np.testing.assert_array_equal(env.agents['agent3'].position, np.array([9, 4]))

def test_some_agents_cannot_move():
    region = 5
    agents = {
        'agent0': WorldAgent(id='agent0', starting_position=np.array([4, 4])),
        'agent1': WorldAgent(id='agent1', starting_position=np.array([0, 4])),
        'agent2': WorldAgent(id='agent2', starting_position=np.array([0, 0])),
        'agent3': WorldAgent(id='agent3', starting_position=np.array([4, 0])),
        'agent4': WorldAgent(id='agent4', starting_position=np.array([2, 2])),
        'agent5': GridWorldMovementAgent(id='agent5', starting_position=np.array([2, 1]), move = 1),
    }
    env = GridWorldMovementEnv(region=region, agents=agents)
    assert 'move'     in env.agents['agent5'].action_space and     isinstance(env.agents['agent5'], GridMovementAgent)
    assert 'move' not in env.agents['agent0'].action_space and not isinstance(env.agents['agent5'], GridMovementAgent)
    assert 'move' not in env.agents['agent1'].action_space and not isinstance(env.agents['agent5'], GridMovementAgent)
    assert 'move' not in env.agents['agent2'].action_space and not isinstance(env.agents['agent5'], GridMovementAgent)
    assert 'move' not in env.agents['agent3'].action_space and not isinstance(env.agents['agent5'], GridMovementAgent)
    assert 'move' not in env.agents['agent4'].action_space and not isinstance(env.agents['agent5'], GridMovementAgent)
