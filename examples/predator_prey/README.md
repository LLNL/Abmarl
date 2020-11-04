# Predator Prey Example

Predator-Prey is a multiagent environment usefuly for exploring competitve behaviors between groups
of agents. For this experiment we want to see how trained predators behave against various
heuristic prey behavior. We will train predator behavior against three heuristic prey policies:
(1) A Random Policy where the prey move around randomly, (2) A Just Sit Policy, where the prey
don't move at all, and a (3) Run Away Policy, where the prey run away from the
predator.

## Installation

In addition the listed dependencies in the [main README](/README.md), the Predator-Prey
environment requires two more packages:
1. Install python-box: `pip install python-box`
1. Install seaborn: `pip install seaborn`

## Creating a configuration script

We start with a Python configuration script that defines the simulation
environment and training agents. This example's scipt is
[predator_prey_training.py](predator_prey_training.py).
To begin, we'll import the Predator-Prey simulation environment, wrap
it with our MultiAgentWrapper, and register it with RLlib.

```python
from admiral.envs.predator_prey import PredatorPrey
from admiral.envs.wrappers import MultiAgentWrapper
from ray.tune.registry import register_env

# Configure the environment
env_name = "PredatorPrey"
region = 6
predators = [{'id': 'predator0', 'view': region-1}]
prey = [{'id': 'prey' + str(i), 'view': region-1} for i in range(7)]
agents = prey + predators
env_config = {'agents': agents, 'region': region}

env, agents = PredatorPrey.build(env_config)
register_env(env_name, lambda env_config: MultiAgentWrapper.wrap(env))
```

Next, we implement the heuristic policies we want to train against.
`RandomPolicy` is a built-in policy, so we only
need to define Just Sit and Run Away. Admiral supplies a `HeuristicPolicy` parent
class, in which only the `compute_action` function needs to be defined.

```python
from admiral.pols import RandomAction, HeuristicPolicy

class JustSit(HeuristicPolicy):
    """A policy that has the agents just sit in the same space"""
    def compute_actions(self, obs_batch, *args, **kwargs):
        return [PredatorPrey.Actions.STAY.value for _ in obs_batch], [], {}

import numpy as np
class RunAwayFromSinglePredatorGridObs(HeuristicPolicy):
    """
    A policy that has the agents run away from the predator. This assumes grid
    observations and assumes there is only one predator.
    """
    def compute_actions(self, obs_batch, *args, **kwargs):
        action = []
        for obs in obs_batch:
            my_loc = int(obs.shape[0] / 2)
            predator_x, predator_y = \
                np.where(obs==PredatorPrey.AgentType.PREDATOR.value)[1][0], \
                np.where(obs==PredatorPrey.AgentType.PREDATOR.value)[0][0]
            if abs(predator_x - my_loc) > abs(predator_y - my_loc): # Move in the y direction
                if my_loc > predator_y: # I am below the predator
                    action.append(PredatorPrey.Actions.MOVE_DOWN.value)
                else:
                    action.append(PredatorPrey.Actions.MOVE_UP.value)
            else: # Move in the x direction
                if my_loc > predator_x: # I am to the right of the predator
                    action.append(PredatorPrey.Actions.MOVE_RIGHT.value)
                else:
                    action.append(PredatorPrey.Actions.MOVE_LEFT.value)
        return action, [], {}
```

With the heuristic policies defined, we can now assign agents to policies. This is done
through a policy mapping function that maps the agents' ids to their
respective policy ids. We'll define each of the three heuristic policies for
the prey, but we'll only assign one in the policy mapping function.

```python
policies = {
    'predator': (None, agents['predator0'].observation_space, agents['predator0'].action_space, {}),
    'prey_sit': (JustSit, agents['prey0'].observation_space, agents['prey0'].action_space, {}),
    'prey_random': (RandomAction, agents['prey0'].observation_space, agents['prey0'].action_space, {}),
    'prey_runaway': (RunAwayFromSinglePredatorGridObs, agents['prey0'].observation_space, agents['prey0'].action_space, {})
}
def policy_mapping_fn(agent_id): # Return the policy id
    if agent_id.startswith('prey'):
        return 'prey_sit'
    else:
        return 'predator'
```

We've built the environment, defined the heuristic policies, and mapped
agent ids to those policies. The last thing to do now is to specifiy which
learning algorithm we will use and to wrap all the settings together into a
single `params` variable. In this experiment, we'll just use vanilla policy
gradient.

```python
algo_name = "PG" 

params = {
    'experiment': { # Experiment details, including the name of the experiment
        'title': '{}'.format('PredatorPrey-justsit'),
    },
    'ray_tune': { # Parameters for launching with RLlib
        'run_or_experiment': algo_name,
        'checkpoint_freq': 50,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 20_000,
        },
        'verbose': 2,
        'config': {
            # --- Environment ---
            'env': env_name,
            'env_config': env_config,
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
In this experiment, we care about the performance of the predator, so we'll
look at the reward associated with the predator policy. This will likely be
located on the second page of the tensorboard page.

![predator_reward](/.images/predator_reward.png)

Having trained the predator to behave when the prey sit still, we now want to
see what the predator learns if the prey move around randomly. To do this,
we simply make one small change in the configuration script

```python
...
def policy_mapping_fn(agent_id):
    if agent_id.startswith('prey'):
        return 'prey_random' # Change the name of the prey policy
    else:
        return 'predator'
...

params = {
    'experiment': { # Change the name of the experiment
        'title': '{}'.format('PredatorPrey-random'),
    },
    ...
```

and call the train command again.

```
admiral train predator_prey_training.py
```
Once this is done training, we can do the same with the runaway policy to
obtain trained behavior in all three experiments.

### Playing
Having successfully trained a predator to hunt prey with various heuristic
behavior, we can vizualize the agent's learned behavior with the `play` command,
which takes as argument the output directory from the training session stored
in `~/admiral_results`. For example, the command

```
admiral play ~/admiral_results/PredatorPrey-runaway-2020-08-25_09-30/ -n 5 --record
```

will load the training session when the prey did not move (notice that the
directory name is the experiment name from the configuration script appended with a
timestamp) and display an animation of 5 episodes. The `--record` flag will
save the animations as `.mp4` videos in the training directory.

### Analyzing
Upon visualizing the trained preator's behavior, we notice that the
predator seems to "herd" the prey into the top left corner of the map, where
it then eats them all. We can further investigate this behavior using the
`analyze` command along with an analysis script.

Analysis scripts implement a `run` command which takes the environment and
the agents as input arguments. At this point, the researcher can define any
script to further investigate the agent's behavior. In this
example, we will craft a script that records how often the agent visits each grid square and how
often it attacks from each grid square. This file is
[movement_map.py](movement_map.py).

```python
def run(env, agent):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create a grid
    grid = np.zeros((env.region, env.region))
    attack = np.zeros((env.region, env.region))

    # Run the trained policy
    policy_agent_mapping = agent.config['multiagent']['policy_mapping_fn']
    for episode in range(100): # Run 100 trajectories
        print('Episode: {}'.format(episode))
        obs = env.reset()
        pox, poy = env.agents['predator0'].position
        grid[pox, poy] += 1
        while True:
            joint_action = {}
            for agent_id, agent_obs in obs.items():
                if agent_id not in env.agents: continue # Don't get actions for dead agents
                policy_id = policy_agent_mapping(agent_id)
                action = agent.compute_action(agent_obs, policy_id=policy_id)
                joint_action[agent_id] = action
            obs, _, done, _ = env.step(joint_action)
            pox, poy = env.agents['predator0'].position
            grid[pox, poy] += 1
            if joint_action['predator0'] == env.Actions.ATTACK.value: # This is the attack action
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
admiral analyze ~/admiral_results/PredatorPrey-runaway-2020-08-25_09-30/ movement_map.py
```

![position frequency](/.images/position_freq.png)
![attack frequency](/.images/attack_freq.png)

The heatmap figures indicate that our intuition was right--the predator tends to
herd the prey to the top left corner and then attack them one by one.

Notice that creating the analysis script required some in-depth knowledge about
the inner workings of the Predator-Prey environment. This will likely be needed
when analyzing most environments you work with.

