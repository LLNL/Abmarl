.. Admiral documentation overview.

Design
======

A reinforcement learning experiment contains two main components: (1) a simulation
environment and (2) learning agents, which contain policies that map observations
to actions. These policies may be hard-coded by the researcher or trained
by the RL algorithm. In Admiral, these two components are specified in a single
Python configuration script. The components can be defined in-script or imported
as modules.

Once these components are setup, they are passed as parameters to RLlib's
tune command, which will launch the RLlib application and begin the training
process. The training process will save checkpoints to an output directory,
from which you can visualize and analyze results. The following diagram
demonstrates this workflow.

![Workflow](.images/workflow.png)

#TODO: more content here, especially talking about the AES (agent-environment simulation)/
ABS nature of the repository and environments.

Creating Agents and Environments
--------------------------------

Using Agent, AgentBasedSimulation, and Managers to construct a simulation ready
for training in Admiral.

.. _abs:

Agent Based Simulation
``````````````````````

.. _sim-man:

Simulation Managers
```````````````````

Training with an Experiment Configuration
-----------------------------------------
We must define a configuration script that specifies the environment and agent
parameters. Once we have this script, we can use the command-line interface
to train, visualize, and analyze agent behavior.

This example demonstrates a simple corridor environment with multiple agents and
can be found [here](/examples/multi_corridor_example.py).

from admiral.envs.corridor import MultiCorridor
from admiral.managers import TurnBasedManager
from admiral.external import MultiAgentWrapper

env = MultiAgentWrapper(AllStepManager(MultiCorridor()))

env_name = "MultiCorridor"
from ray.tune.registry import register_env
register_env(env_name, lambda env_config: env)

agents = env.unwrapped.agents
policies = {
    'corridor': (None, agents['agent0'].observation_space, agents['agent0'].action_space, {})
}
def policy_mapping_fn(agent_id):
    return 'corridor'

# Experiment parameters
params = {
    'experiment': {
        'title': f'{env_name}',
        'env_creator': lambda config=None: env,
    },
    'ray_tune': {
        'run_or_experiment': 'PG',
        'checkpoint_freq': 50,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 20_000,
        },
        'verbose': 2,
        'config': {
            # --- Environment ---
            'env': env_name,
            'horizon': 200,
            'env_config': {},
            # --- Multiagent ---
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_mapping_fn,
            },
            # --- Parallelism ---
            "num_workers": 7,
            "num_envs_per_worker": 1,
        },
    }
}

**Warning**: This example has `num_workers` set to 7 because we are on a computer
with 8 CPU's. You may need to adjust this for your computer to be `<cpu count> - 1`.


With the configuration scipt complete, we can utilize the command line interface
to train our agents. We simply type

```
admiral train multi_corridor_example.py
```
where `multi_corridor_example.py` is the name of our script. This will launch
Admiral, which will process the script and launch RLlib according to the
specified parameters. This particular example should take 1-10 minutes to
train, depending on your compute capabilities. You can view the performance in real time in tensorboard with
```
tensorboard --logdir ~/admiral_results
```


Visualizing
-----------
We can vizualize the agents' learned behavior with the `visualize` command, which
takes as argument the output directory from the training session stored in `~/admiral_results`. For example, the command

```
admiral visualize ~/admiral_results/MultiCorridor-2020-08-25_09-30/ -n 5 --record
```

will load the training session (notice that the directory name is the experiment
name from the configuration script appended with a timestamp) and display an animation
of 5 episodes. The `--record` flag will save the animations as `.mp4` videos in
the training directory.

Visualizing the trained behavior, we can see that all the agents learn to move
to the right, which is the desired behavior.

Analyzing
---------

See the [Predator-Prey example](examples/predator_prey), which provides a great use case
for analyzing agent behaviors.

## Running at scale with HPC
See the [magpie example](examples/magpie/), which provides a walkthrough
for launching a training experiment on multiple compute nodes.

.. _external:

External Integration
--------------------

Some text about how we integrate with gym and marl envs.



