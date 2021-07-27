.. Abmarl documentation Gym tutorial.

.. _tutorial_gym:

Gym Environment
===============

Abmarl can be used with OpenAI Gym environments. In this tutorial, we'll create
a training configuration file that trains a gym environment. This tutorial uses
the `gym configuration <https://github.com/LLNL/Abmarl/blob/main/examples/gym_example.py>`_.


Training a Gym Environment
--------------------------

Simulation Setup
````````````````

We'll start by creating gym's built-in guessing game.

.. code-block:: python

   import gym
   from ray.tune.registry import register_env

   sim = gym.make('GuessingGame-v0')
   sim_name = "GuessingGame"
   register_env(sim_name, lambda sim_config: sim)

.. NOTE::

   Even gym's built-in environments need to be registered with RLlib.

Experiment Parameters
`````````````````````

All training configuration parameters are stored in a dictionary called `params`.
Having setup the simualtion, we can now create the `params` dictionary that will
be read by Abmarl and used to launch RLlib.

.. code-block:: python

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


Command Line interface
``````````````````````
With the configuration file complete, we can utilize the command line interface
to train our agents. We simply type ``abmarl train gym_example.py``,
where `gym_example.py` is the name of our configuration file. This will launch
Abmarl, which will process the file and launch RLlib according to the
specified parameters. This particular example should take 1-10 minutes to
train, depending on your compute capabilities. You can view the performance
in real time in tensorboard with ``tensorboard --logdir ~/abmarl_results``.
