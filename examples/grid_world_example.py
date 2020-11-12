
from matplotlib import pyplot as plt
import numpy as np

from admiral.component_envs.world import WorldAgent
from admiral.component_envs.movement import GridMovementEnv

env = GridMovementEnv(
    region=10,
    agents={f'agent{i}': WorldAgent(id=f'agent{i}') for i in range(4)}
)
env.reset()
fig = plt.gcf()
env.render(fig=fig)

for _ in range(20):
    fig.clear()
    for agent in env.agents.values():
        env.process_move(agent, np.random.randint(-1, 2, size=(2,)))
    env.render(fig=fig)
    plt.plot()
    plt.pause(1e-17)
