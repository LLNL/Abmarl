
from matplotlib import pyplot as plt

from admiral.component_envs import GridWorldEnv
from admiral.component_envs import WorldAgent

env = GridWorldEnv(
    region=10,
    agents={f'agent{i}': WorldAgent(id=f'agent{i}') for i in range(4)}
)
env.reset()
env.render()

plt.show()
