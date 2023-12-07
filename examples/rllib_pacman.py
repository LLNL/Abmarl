
from abmarl.examples.sim.pacman import PacmanSim, PacmanAgent, WallAgent
from abmarl.managers import AllStepManager


object_registry = {
    'P': lambda n: PacmanAgent(
        id='pacman',
        encoding=1,
        # view_range=2,
        render_color='yellow',
    ),
    'W': lambda n: WallAgent(
        id=f'wall_{n}',
        encoding=2,
        # blocking=True,
        render_shape='s'
    )
}
file_name = '/g/g13/rusu1/abmarl/examples/pacman_simple.txt'
sim = AllStepManager(
    PacmanSim.build_sim_from_file(
        file_name,
        object_registry,
        states={'PositionState', 'OrientationState'},
        dones={'ActiveDone'},
        observers={'PositionCenteredEncodingObserver'}
    )
)


from abmarl.trainers import DebugTrainer
# Setup the Debugger
debugger = DebugTrainer(
    sim=sim,
    output_dir="output_dir",
    name="Pacman_simple_demo"
)
debugger.train(iterations=4, render=True, horizon=200)