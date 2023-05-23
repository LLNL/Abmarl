
from abmarl.examples import MultiMazeNavigationAgent, MultiMazeNavigationSim
from abmarl.sim.gridworld.agent import GridWorldAgent
from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper

agents = {
    'target': GridWorldAgent(id='target', encoding=1, render_color='g'),
    **{
        f'barrier{i}': GridWorldAgent(
            id=f'barrier{i}',
            encoding=2,
            render_shape='s',
            render_color='gray',
        ) for i in range(20)
    },
    **{
        f'navigator{i}': MultiMazeNavigationAgent(
            id=f'navigator{i}',
            encoding=3,
            render_color='b',
            view_range=5
        ) for i in range(5)
    }
}

sim = MultiAgentWrapper(
    AllStepManager(
        MultiMazeNavigationSim.build_sim(
            10, 10,
            agents=agents,
            overlapping={1: {3}, 3: {3}},
            target_agent=agents['target'],
            barrier_encodings={2},
            free_encodings={1, 3},
            cluster_barriers=True,
            scatter_free_agents=True,
            no_overlap_at_reset=True
        )
    )
)


sim_name = "MultiMazeNavigation"
from ray.tune.registry import register_env
register_env(sim_name, lambda sim_config: sim)


policies = {
    'navigator': (
        None,
        sim.sim.agents['navigator0'].observation_space,
        sim.sim.agents['navigator0'].action_space,
        {}
    )
}


def policy_mapping_fn(agent_id):
    return 'navigator'


# Experiment parameters
params = {
    'experiment': {
        'title': f'{sim_name}',
        'sim_creator': lambda config=None: sim,
    },
    'ray_tune': {
        'run_or_experiment': 'A2C',
        'checkpoint_freq': 50,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 2000,
        },
        'verbose': 2,
        'config': {
            # --- Simulation ---
            'disable_env_checking': False,
            'env': sim_name,
            'horizon': 200,
            'env_config': {},
            # --- Multiagent ---
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_mapping_fn,
            },
            # "lr": 0.0001,
            # --- Parallelism ---
            # Number of workers per experiment: int
            "num_workers": 7,
            # Number of simulations that each worker starts: int
            "num_envs_per_worker": 1, # This must be 1 because we are not "threadsafe"
        },
    }
}
