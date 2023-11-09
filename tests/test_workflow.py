
import os

from abmarl.debug import debug
from abmarl.train import train
from abmarl.stage import visualize, analyze

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
        'checkpoint_freq': 5,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 2,
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
            # --- Parallelism ---
            # Number of workers per experiment: int
            "num_workers": 7,
            # Number of simulations that each worker starts: int
            "num_envs_per_worker": 1, # This must be 1 because we are not "threadsafe"
        },
    }
}

def tally_rewards(sim, trainer):
    """
    Analyze the behavior of your trained policies using the simulation and trainer
    from your RL experiment.

    Args:
        sim:
            Simulation Manager object from the experiment.
        trainer:
            Trainer that computes actions using the trained policies.
    """
    # Run the simulation with actions chosen from the trained policies
    policy_agent_mapping = trainer.config['multiagent']['policy_mapping_fn']
    for episode in range(5):
        episode_reward = 0
        print('Episode: {}'.format(episode))
        obs = sim.reset()
        done = {agent: False for agent in obs}
        while True: # Run until the episode ends
            # Get actions from policies
            joint_action = {}
            for agent_id, agent_obs in obs.items():
                if done[agent_id]: continue # Don't get actions for done agents
                policy_id = policy_agent_mapping(agent_id)
                action = trainer.compute_action(agent_obs, policy_id=policy_id)
                joint_action[agent_id] = action
            # Step the simulation
            obs, reward, done, info = sim.step(joint_action)
            episode_reward += sum(reward.values())
            if done['__all__']:
                break
        print(episode_reward)


# --- Demonstrate Workflow --- #


def test_debug(tmpdir):
    output_dir = debug(params)
    assert output_dir == params['ray_tune']['local_dir']
    assert os.path.exists(output_dir)
    assert "Episode_0_by_event.txt" in os.listdir(output_dir)
    assert "Episode_0_by_agent.txt" in os.listdir(output_dir)

    with open(os.path.join(output_dir, "Episode_0_by_event.txt"), 'r') as event_file:
        text = event_file.read()
        assert "Reset" in text
        assert "Observation" in text
        assert "Step 0" in text
        assert "Action" in text
        assert "Reward" in text
        assert "Done" in text

    with open(os.path.join(output_dir, "Episode_0_by_agent.txt"), 'r') as agent_file:
        text = agent_file.read()
        assert "Observations" in text
        assert "Actions" in text
        assert "Rewards" in text
        assert "Dones" in text

    output_dir = debug(params, episodes=11, steps_per_episode=10)
    assert output_dir == params['ray_tune']['local_dir']
    assert os.path.exists(output_dir)
    assert len(os.listdir(output_dir)) == 22


import ray
ray.init()

print("\n\n\n# --- Debugging --- #\n\n\n")
debug(params) # Debug the simulation with random policies
print("\n\n\n# --- Training --- #\n\n\n")
output_dir = train(params) # Train the policies with RLlib
print("\n\n\n# --- Visualizing --- #\n\n\n")
visualize(params, output_dir) # Visualize the trained policies in the simulation
print("\n\n\n# --- Analyzing --- #\n\n\n")
analyze(params, output_dir, tally_rewards) # Analyze the trained policies in the simulation

ray.shutdown()
