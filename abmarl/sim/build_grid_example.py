
from pprint import pprint

from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim.simplified_grid import WallAgent, ExploringAgent, build_grid_sim
from abmarl.sim import ActingAgent, ObservingAgent

object_registry = {
    'A': lambda n: ExploringAgent(id=f'explorer{n}', view_range=3, move_range=1),
    'W': lambda n: WallAgent(id=f'wall{n}')
}

file_name = 'starting_grid.txt'

fig = plt.figure()

sim = build_grid_sim(object_registry, file_name)
sim.reset()
sim.render(fig=fig)

plt.show()
