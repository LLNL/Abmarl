
import os
import pytest
from ray.tune.registry import register_env

from abmarl.debug import debug
from abmarl.train import train
from abmarl.stage import visualize, analyze

from abmarl.examples import MultiMazeNavigationAgent, MultiMazeNavigationSim
from abmarl.sim.gridworld.agent import GridWorldAgent
from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper


def pytest_namespace():
    return {'output_dir': None}


def pytest_collection_modifyitems(items):
    items = ["test_debug", "test_train", "test_visualize", "test_analyze"]
    return items


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
        'checkpoint_freq': 2,
        'checkpoint_at_end': False,
        'stop': {
            'episodes_total': 100,
        },
        'verbose': 0,
        'config': {
            # --- Simulation ---
            'disable_env_checking': False,
            'env': sim_name,
            'horizon': 100,
            'env_config': {},
            # --- Multiagent ---
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_mapping_fn,
            },
            # --- Parallelism ---
            # Number of workers per experiment: int
            "num_workers": 0,
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
    from abmarl.managers import SimulationManager
    assert isinstance(sim, SimulationManager), "sim must be a SimulationManager."
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


def test_debug():
    output_dir = debug(params)
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
    assert os.path.exists(output_dir)
    assert len(os.listdir(output_dir)) == 22


def test_train():
    import ray
    ray.init()
    output_dir = train(params)
    pytest.output_dir = output_dir
    assert os.path.exists(output_dir)
    assert len(os.listdir(output_dir)) == 1
    a2c_dir = os.path.join(output_dir, "A2C")
    assert len(os.listdir(a2c_dir)) == 3
    sub_dir = ''
    for name in os.listdir(a2c_dir):
        if name.startswith("A2C_"):
            sub_dir = name
            break
    sub_dir = os.path.join(a2c_dir, sub_dir)
    assert "checkpoint_000002" in os.listdir(sub_dir)
    assert "checkpoint_000004" in os.listdir(sub_dir)
    # Hard to determine just how many checkpoints will be produced, so we will
    # go with at least 2


def test_visualize():
    output_dir = pytest.output_dir
    visualize(params, output_dir, checkpoint=5, episodes=3, steps_per_episode=20, record_only=True)
    assert 'Episode_0.gif' in os.listdir(output_dir)
    assert 'Episode_1.gif' in os.listdir(output_dir)
    assert 'Episode_2.gif' in os.listdir(output_dir)


def test_analyze():
    import ray
    output_dir = pytest.output_dir
    analyze(params, output_dir, tally_rewards, checkpoint=10)
    ray.shutdown()
