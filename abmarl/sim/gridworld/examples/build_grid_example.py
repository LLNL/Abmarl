
from matplotlib import pyplot as plt

from abmarl.sim.gridworld.examples.gridworld_example import WallAgent, FoodAgent, GridSim

object_registry = {
    'A': lambda n: FoodAgent(id=f'explorer{n}', initial_health=1, encoding=1),
    'W': lambda n: WallAgent(id=f'wall{n}', encoding=1, render_shape='X')
}

file_name = 'starting_grid.txt'

fig = plt.figure()

sim = GridSim.build_sim_from_file(file_name, object_registry, attack_mapping={})
sim.reset()
sim.render(fig=fig)

plt.show()
