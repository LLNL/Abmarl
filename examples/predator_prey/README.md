# Predator Prey Example

Predator-Prey is a multiagent environment usefuly for exploring competitve behaviors between groups
of agents. We will train both predator and prey behavior using MARL.

## Installation

In addition the listed dependencies in the [main README](/README.md), the Predator-Prey
environment requires seaborn:
1. Install seaborn: `pip install seaborn`

## Creating a configuration script

We start with a Python configuration script that defines the simulation
environment and training agents. This example's scipt is
[predator_prey_training.py](predator_prey_training.py).
To begin, we'll import the Predator-Prey simulation environment, wrap
it with our MultiAgentWrapper, and register it with RLlib.

```python
from admiral.envs.examples.predator_prey import PredatorPreyEnv, Predator, Prey
from admiral.managers import AllStepManager
from admiral.envs.wrappers import MultiAgentWrapper
from ray.tune.registry import register_env

# Configure the environment
region = 6
predators = [Predator(id=f'predator{i}', attack=1) for i in range(2)]
prey = [Prey(id=f'prey{i}') for i in range(7)]
agents = predators + prey
env_config = {
    'region': region,
    'max_steps': 200,
    'agents': agents,
}
env_name = 'PredatorPrey'

env = MultiAgentWrapper(AllStepManager(PredatorPreyEnv.build(env_config)))
agents = env.unwrapped.agents
register_env(env_name, lambda env_config: env)
```

Next, we assign agents to policies. This is done
through a policy mapping function that maps the agents' ids to their
respective policy ids. This experiment has two policies, one that is shared by
the predators and another that is shared by the prey.

```python
policies = {
    'predator': (None, agents['predator0'].observation_space, agents['predator0'].action_space, {}),
    'prey': (None, agents['prey0'].observation_space, agents['prey0'].action_space, {})
}
def policy_mapping_fn(agent_id):
    if agent_id.startswith('prey'):
        return 'prey'
    else:
        return 'predator'
```

The last thing to do now is to specifiy which
learning algorithm we will use and to wrap all the settings together into a
single `params` variable. In this experiment, we'll just use vanilla policy
gradient.

```python
algo_name = "PG" 

params = {
    'experiment': { # Experiment details, including the name of the experiment
        'title': '{}'.format('PredatorPrey'),
        'env_creator': lambda config=None: env,
    },
    'ray_tune': { # Parameters for launching with RLlib
        'run_or_experiment': algo_name,
        'checkpoint_freq': 50,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 2000,
        },
        'verbose': 2,
        'config': {
            # --- Environment ---
            'env': env_name,
            'env_config': env_config,
            'horizon': 200,
            # --- Multiagent ---
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_mapping_fn,
            },
            # --- Parallelism ---
            "num_workers": 7, # May need to adjust depending on available CPU resources
            "num_envs_per_worker": 1 # Leave this at 1 because our environment is not copy-safe
        },
    }
}
```

**Warning**: This example has `num_workers` set to 7 because we are on a computer
with 8 CPU's. You may need to adjust this for your computer to be `<cpu count> - 1`.

## Using the command line 

### Training

With the [configuration scipt complete](predator_prey_training.py),
we can utilize the command line interface to train our predator. We simply type

```
admiral train predator_prey_training.py
```
This will launch Admiral, which will process the script and launch RLlib according to the
specified parameters. This particular example should take about 10 minutes to
train. You can view the performance in real time in tensorboard with
```
tensorboard --logdir ~/admiral_results
```
We can find the rewards associated with the policies on the second page of tensorboard.


### Visualizing
Having successfully trained predators to attack prey, we can vizualize the agents'
learned behavior with the `visualize` command,
which takes as argument the output directory from the training session stored
in `~/admiral_results`. For example, the command

```
admiral visualize ~/admiral_results/PredatorPrey-2020-08-25_09-30/ -n 5 --record
```

will load the training session (notice that the
directory name is the experiment name from the configuration script appended with a
timestamp) and display an animation of 5 episodes. The `--record` flag will
save the animations as `.mp4` videos in the training directory.

### Analyzing
We can further investigate the learned
behaviors using the `analyze` command along with an analysis script.

Analysis scripts implement a `run` command which takes the environment and
the training agent as input arguments. At this point, the researcher can define any
script to further investigate the agents' behavior. In this
example, we will craft a script that records how
often a predator attacks from each grid square. This file is
[movement_map.py](movement_map.py).

```python
def run(env, agent):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    sim = env.unwrapped

    # Create a grid
    grid = np.zeros((sim.env.region, sim.env.region))
    attack = np.zeros((sim.env.region, sim.env.region))

    # Run the trained policy
    policy_agent_mapping = agent.config['multiagent']['policy_mapping_fn']
    for episode in range(100): # Run 100 trajectories
        print('Episode: {}'.format(episode))
        obs = sim.reset()
        done = {agent: False for agent in obs}
        pox, poy = sim.agents['predator0'].position
        grid[pox, poy] += 1
        while True:
            joint_action = {}
            for agent_id, agent_obs in obs.items():
                if done[agent_id]: continue # Don't get actions for dead agents
                policy_id = policy_agent_mapping(agent_id)
                action = agent.compute_action(agent_obs, policy_id=policy_id)
                joint_action[agent_id] = action
            obs, _, done, _ = sim.step(joint_action)
            pox, poy = sim.agents['predator0'].position
            grid[pox, poy] += 1
            if joint_action['predator0']['attack'] == 1: # This is the attack action
                attack[pox, poy] += 1
            if done['__all__']:
                break
    
    plt.figure(1)
    plt.title("Position concentration")
    ax = sns.heatmap(np.flipud(np.transpose(grid)), linewidth=0.5)

    plt.figure(2)
    plt.title("Attack action frequency")
    ax = sns.heatmap(np.flipud(np.transpose(attack)), linewidth=0.5)

    plt.show()
```

We can run it with

```
admiral analyze ~/admiral_results/PredatorPrey-2020-08-25_09-30/ movement_map.py
```

![attack frequency](/.images/attack_freq.png)

The heatmap figures indicate that the predator spends most of its time attacking
prey from the center of the map and rarely ventures to the corners.

Notice that creating the analysis script required some in-depth knowledge about
the inner workings of the Predator-Prey environment. This will likely be needed
when analyzing most environments you work with.
