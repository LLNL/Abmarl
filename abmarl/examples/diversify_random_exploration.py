
from copy import deepcopy

from matplotlib import pyplot as plt
import numpy as np

from abmarl.examples.sim.diversify import BlankSpace, Apple, Peach, Pear, Plum, Cherry, DiversifySim, FutureTree

agents = {
    **{
        f'blank0{n}': BlankSpace(
            id=f'blank0{n}',
            initial_position = np.array([2 * n + 1, 0])
        ) for n in range(12)
    },
    **{
        f'blank1{n}': BlankSpace(
            id=f'blank1{n}',
            initial_position = np.array([2 * n, 1])
        ) for n in range(12)
    },
    **{
        f'blank2{n}': BlankSpace(
            id=f'blank2{n}',
            initial_position = np.array([2 * n + 1, 2])
        ) for n in range(12)
    },
    **{ # 4th column
        f'blank3{n}': BlankSpace(
            id=f'blank3{n}',
            initial_position = np.array([2 * n, 3])
        ) for n in range(12)
    },
    'crab_apple': Apple(
        id='crab_apple',
        initial_position=np.array([13, 1]),
        render_shape='d'
    ),
    **{
        f'apple{n}': Apple(id=f'apple{n}') for n in range(15)
    },
    **{
        f'peach{n}': Peach(id=f'peach{n}') for n in range(4)
    },
    **{
        f'plum{n}': Plum(id=f'plum{n}') for n in range(4)
    },
    **{
        f'pear{n}': Pear(id=f'pear{n}') for n in range(4)
    },
    **{
        f'cherry{n}': Cherry(id=f'cherry{n}') for n in range(4)
    },
    **{
        f'future{n}': FutureTree(id=f'future{n}') for n in range(16)
    },
}

reward_type = 'neighbor'
sim = DiversifySim.build_sim(
    24, 4, agents=agents,
    reward_type='neighbor',
    # reward_type='neighbor2',
    # reward_type='distance',
)

for trial in range(3): # Number of trials
    min_reward = 10
    minimizing_sim = None
    for _ in range(1000):
        sim.reset()
        reward = sim.get_reward()
        if reward < min_reward:
            min_reward = reward
            minimizing_sim = deepcopy(sim)

    fig = plt.figure(figsize=(8, 10), dpi=80)
    minimizing_sim.render(fig=fig)
    fig.savefig(f'{reward_type}_t-{trial}_r-{min_reward:.3f}.png', dpi=80)
