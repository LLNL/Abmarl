
import numpy as np

from abmarl.sim.gridworld.grid import Grid
from abmarl.sim.gridworld.actor import MovingAgent

grid = Grid(5,6)

moving_agents = {
    'agent0': MovingAgent(
        id='agent0', initial_position=np.array([3, 4]), encoding=1, move_range=1
    ),
    'agent1': MovingAgent(
        id='agent1', initial_position=np.array([2, 2]), encoding=2, move_range=2
    ),
    'agent2': MovingAgent(
        id='agent2', initial_position=np.array([0, 1]), encoding=1, move_range=1
    ),
    'agent3': MovingAgent(
        id='agent3', initial_position=np.array([3, 1]), encoding=3, move_range=3
    ),
}
