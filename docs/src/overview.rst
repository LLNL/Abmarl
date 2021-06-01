.. Admiral documentation overview.

Design
======

A reinforcement learning experiment contains two main components: (1) a simulation
environment and (2) a trainer, which contain policies that map observations
to actions. These policies may be hard-coded by the researcher or trained
by the RL algorithm. In Admiral, these two components are specified in a single
Python configuration file. The components can be defined in-file or imported
as modules.

Once these components are set up, they are passed as parameters to RLlib's
tune command, which will launch the RLlib application and begin the training
process. The training process will save checkpoints to an output directory,
from which you can visualize and analyze results. The following diagram
demonstrates this workflow.

.. image:: .images/workflow.png
  :width: 800
  :alt: Admiral Workflow


Creating Agents and Environments
--------------------------------

Admiral provides three interfaces for setting up an agent-based simulation environment.

.. _overview_agent:

Agent
`````

.. ATTENTION::
   TODO: The link for Agent API is not very good because it is an organizational
   class that inherits from Observing and Acting Agent. We should come up with a
   better link/structure of the API.

First, we have :ref:`Agents <api_agent>`. An agent is an object with an observation and
action space. Many practitioners may be accustomed to gym.Env's interface, which
defines the observation and action space for the *environment*. However, in heterogeneous
multi-agent settings, each *agent* can have different spaces; thus we assign these
spaces to the agents and not the environment.

An agent can be created like so:

.. code-block:: python

   from gym.spaces import Discrete, Box
   from admiral.envs import Agent
   agent = Agent(
       id='agent0',
       observation_space=Box(-1, 1, (2,)),
       action_space=Discrete(3)
   )

At this level, the Agent is basically a dataclass. We have left it open for our
users to extend its features as they see fit.

.. _abs:

Agent Based Simulation
``````````````````````
Next, we define an :ref:`Agent Based Simulation <api_abs>`, or ABS for short, with the
ususal ``reset`` and ``step``
functions that we are used to seeing in RL environments. These functions, however, do
not return anything; the state information must be obtained from the getters:
``get_obs``, ``get_reward``, ``get_done``, ``get_all_done``, and ``get_info``. The getters
take an agent's id as input and return the respective information from the simulation's
state. The ABS also contains a dictionary of agents that "live" in the environment.

An Agent Based Simulation can be created and used like so:

.. code-block:: python

   from admiral.envs import Agent, AgentBasedSimulation   
   class MySim(AgentBasedSimulation):
       def __init__(self, agents=None, **kwargs):
           self.agents = agents
           ... # Implement the ABS interface

   # Create a dictionary of agents
   agents = {f'agent{i}': Agent(id=f'agent{i}', ...) for i in range(10)}
   # Create the ABS environment with the agents
   env = MySim(agents=agents)
   env.reset()
   # Get the observations
   obs = {agent.id: env.get_obs(agent.id) for agent in agents.values()}
   # Take some random actions
   env.step({agent.id: agent.action_space.sample() for agent in agents.values()})
   # See the reward for agent3
   print(env.get_reward('agent3'))

.. IMPORTANT::
   Your implementation of AgentBasedSimulation should call ``finalize`` at the
   end of its ``__init__``.
   Finalize ensures that all agents are configured and ready to be used for training.

.. _sim-man:

Simulation Managers
```````````````````

The Agent Based Simulation interface does not specify an ordering for agents' interactions
with the environment. This is left open to give our users maximal flexibility. However,
in order to interace with RLlib's learning library, we provide :ref:`Simulation Managers <api_sim>`
which specify the output from ``reset`` and ``step`` as RLlib expects it. Specifically,
1. Agents that appear in the output dictionary (from reset or step) will provide
actions at the next step, and
2. Agents that are done on this step will not provide actions on the next step.

Simulation managers are open-ended requiring only ``reset`` and ``step`` with output
described above. For convenience, we have provided two managers: :ref:`Turn Based <api_turn_based>`,
which implements turn-based games; and :ref:`All Step <api_all_step>`, which has every non-done
agent provide actions at each step.

Simluation Managers "wrap" environments, and they can be used like so:

.. code-block:: python

   from admiral.managers import AllStepManager
   from admiral.envs import AgentBasedSimulation, Agent
   class MySim(AgentBasedSimulation):
       ... # Define some simulation environment

   # Instatiate the environment
   env = MySim(agents=...)
   # Wrap the environment with the simulation manager
   sim = AllStepManager(env)
   # Get the observations for all agents
   obs = sim.reset()
   # Get simulation state for all non-done agents, regardless of which agents
   # actually contribute an action.
   obs, rewards, dones, infos = sim.step({'agent0': 4, 'agent2': [-1, 1]})


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



