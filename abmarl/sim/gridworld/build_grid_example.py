
from matplotlib import pyplot as plt

from abmarl.sim.gridworld import build_grid_sim
from abmarl.sim.gridworld.gridworld_example import WallAgent, ExploringAgent, GridSim

object_registry = {
    'A': lambda n: ExploringAgent(id=f'explorer{n}', view_range=3, move_range=1),
    'W': lambda n: WallAgent(id=f'wall{n}')
}

file_name = 'starting_grid.txt'

fig = plt.figure()

config = build_grid_sim(file_name, object_registry)
sim = GridSim(**config)
sim.reset()
sim.render(fig=fig)

plt.show()
