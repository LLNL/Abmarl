# ------------------------------------- #
# --- DSSI 2021 RL Class Exercise 4 --- #
# ------------------------------------- #

# In this exercise, we will use ABMARL to train the forager to forage any grid.
# We will initialize the simulation, specify the training parameters,
# and connect the simulation with the trainer. We will run a training experiment
# and then visualize the results and output some training statistics.


# Import the simulation environment and the agents
from abmarl.sim.examples import DeepForagingSim, Forager

# Instatiate the forager
agents = {'forager': Forager(
    id='forager',
    agent_view=3,
    attack_range=1,
)}

# Instatiate the simulation
sim = DeepForagingSim(
    12,
    region=20,
    agents=agents
)

# Prepare the environment for use with RLlib
from abmarl.managers import AllStepManager
from abmarl.external import GymWrapper
sim = GymWrapper(AllStepManager(sim))

from ray.tune.registry import register_env
sim_name = "Foraging"
register_env(sim_name, lambda config: sim)

# Parameters
params = {
    'experiment': {
        'title': '{}'.format(sim_name),
        'sim_creator': lambda config=None: sim,
    },
    'ray_tune': {
        'run_or_experiment': 'A2C',
        'checkpoint_freq': 10,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 2000,
        },
        'verbose': 2,
        'config': {
            # --- Environment ---
            'env': sim_name,
            'horizon': 200,
            'env_config': {},
            # --- Parallel ---
            "num_workers": 0,
            "num_envs_per_worker": 1, # This must be 1 because we are not "threadsafe"
            # "rollout_fragment_length": 200,
            # "batch_mode": "complete_episodes",
            # "train_batch_size": 1000,
        },
    }
}

