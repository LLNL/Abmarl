import gym
from ray.tune.registry import register_env

sim = gym.make('GuessingGame-v0')
sim_name = "GuessingGame"
register_env(sim_name, lambda sim_config: sim)


# Experiment parameters
params = {
    'experiment': {
        'title': f'{sim_name}',
        'sim_creator': lambda config=None: sim,
    },
    'ray_tune': {
        'run_or_experiment': 'A2C',
        'checkpoint_freq': 1,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 2000,
        },
        'verbose': 2,
        'config': {
            # --- Simulation ---
            'env': sim_name,
            'horizon': 200,
            'env_config': {},
            # --- Parallelism ---
            # Number of workers per experiment: int
            "num_workers": 6,
            # Number of simulations that each worker starts: int
            "num_envs_per_worker": 1,
        },
    }
}
