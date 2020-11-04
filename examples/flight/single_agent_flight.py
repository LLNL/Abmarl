""" This example demonstrates training a single agent environment."""

from admiral.envs.flight import build_flight_v0

env_name = 'Flight-v0'
env_config = {
    'birds': 4,
}

from ray.tune.registry import register_env
register_env(env_name, lambda env_config: build_flight_v0(env_config))

algo_name = 'PPO'

params = {
    'experiment': {
        'title': '{}'.format('Flight-single-agent'),
    },
    'ray_tune': {
        'run_or_experiment': algo_name,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 20,
            # 'episodes_total': 20_000,
        },
        'verbose': 2,
        'config': {
            # --- Worker ---
            # 'seed': 24,
            # --- Environment ---
            'env': env_name,
            'env_config': env_config,
            # "horizon": 50,
            # --- Model ---
            "model": {
                'fcnet_activation': 'relu',
                'fcnet_hiddens': [256, 128, 64, 32],
            },
            # "lr": 0.0001,
            # --- Parallelism ---
            # Number of workers per experiment: int
            "num_workers": 7,
            # Number of environments that each worker starts: int
            "num_envs_per_worker": 4,
        },
    }
}