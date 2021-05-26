# Admiral

Admiral is a package for developing agent based simulations and training them
with multiagent reinforcement learning. We provide an intuitive command line
interface for training, visualizing, and analyzing agent behavior. We define an
[Agent Based Simulation interface](/admiral/envs/agent_based_simulation.py) and
[Simulation Managers](/admiral/managers/), which control which agents interact
with the environment at each step. We support integration with several popular
environment interfaces, including [gym.Env](/admiral/external/gym_env_wrapper.py) and
[MultiAgentEnv](/admiral/external/rllib_multiagentenv_wrapper.py).

Admiral is a layer in the Reinforcement Learning stack that sits on top of RLlib.
We leverage RLlib's framework for training agents and extend it to more easily
support custom environments, algorithms, and policies. We enable researchers to
rapidly prototype RL experiments and environment design and lower the barrier
for pre-existing projects to prototype RL as a potential solution.

## Design
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


## Installation

### Simple Installation
Install from requirements file. This uses tensorflow and installs all you need
to run the examples.
1. Install the requirements: `pip install -r requirements.txt`
1. Install Admiral `pip install .` or `pip install -e .`


### Detailed installation
Install each package as needed.

To train:
1. Install tensorflow or pytorch
1. Install ray rllib v1.2.0: `pip install ray[rllib]==1.2.0`
1. Install Admiral: `pip install .` or `pip install -e .`

To visualize:
1. Install matplotlib: `pip install matplotlib`

To run Predator-Prey example:
1. Install seaborn: `pip install seaborn`


## Usage

We must define a configuration script that specifies the environment and agent
parameters. Once we have this script, we can use the command-line interface
to train, visualize, and analyze agent behavior. Full examples can be found
[here](examples/).

### Creating a configuration script

This example demonstrates a simple corridor environment with multiple agents and
can be found [here](/examples/multi_corridor_example.py).

```python
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

```

**Warning**: This example has `num_workers` set to 7 because we are on a computer
with 8 CPU's. You may need to adjust this for your computer to be `<cpu count> - 1`.

### Using the command line 

#### Training

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

#### Visualizing
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

#### Analyzing

See the [Predator-Prey example](examples/predator_prey), which provides a great use case
for analyzing agent behaviors.

## Running at scale with HPC
See the [magpie example](examples/magpie/), which provides a walkthrough
for launching a training experiment on multiple compute nodes.

## Continued Support

Admiral was funded as an ISCP project through Computing's Idea Day call. What you
see here is a minimal viable product (MVP), with enhancements planned in future
ISCP projects. However, this project is meant to be primarily *community driven*.
If you use Admiral in your workflow, please consider contributing any features
you think would be useful to the greater RL community. In addition, please contact
me or fill out a "new issue" if you encounter any bugs or want features to be
implemented.

## Contact

* Edward Rusu, rusu1@llnl.gov
* Ruben Glatt, glatt1@llnl.gov

## Release

LLNL-CODE-815883

