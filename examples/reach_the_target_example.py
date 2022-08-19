
import numpy as np

from abmarl.examples import ReachTheTargetSim, RunningAgent, TargetAgent, BarrierAgent
from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper

grid_size = 7
corners = [
    np.array([0, 0], dtype=int),
    np.array([grid_size - 1, 0], dtype=int),
    np.array([0, grid_size - 1], dtype=int),
    np.array([grid_size - 1, grid_size - 1], dtype=int),
]
agents = {
    **{
        f'barrier{i}': BarrierAgent(
            id=f'barrier{i}'
        ) for i in range(10)
    },
    **{
        f'runner{i}': RunningAgent(
            id=f'runner{i}',
            move_range=2,
            view_range=int(grid_size / 2),
            initial_health=1,
            initial_position=corners[i]
        ) for i in range(4)
    },
    'target': TargetAgent(
        view_range=grid_size,
        attack_range=1,
        attack_strength=1,
        attack_accuracy=1,
        initial_position=np.array([int(grid_size / 2), int(grid_size / 2)], dtype=int)
    )
}
overlapping = {
    2: [3],
    3: [2, 3]
}
attack_mapping = {
    2: [3]
}

sim = MultiAgentWrapper(
    AllStepManager(
        ReachTheTargetSim.build_sim(
            7, 7,
            agents=agents,
            overlapping=overlapping,
            attack_mapping=attack_mapping
        )
    )
)

from abmarl.trainers import DebugTrainer
trainer = DebugTrainer(sim=sim.sim, output_dir='abmarl_results')
trainer.train(render=True, horizon=24)

# sim_name = "ReachTheTarget"
# from ray.tune.registry import register_env

