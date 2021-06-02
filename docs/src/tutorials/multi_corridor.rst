.. Admiral documentation MultiCorridor tutorial.

.. _tutorial_multi_corridor:

MultiCorridor
=============

MultiCorridor is derived from RLlib's `simple corridor <https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py#L65>`_,
wherein agents must learn to move to the right to reach the end of the corridor.
Our implementation provides the ability to instantiate multiple agents in the simulation
and restricts agents from occupying the same square. Every agent is homogeneous:
they all have the same action space, observation space, and objective function.

.. image:: /.images/multicorridor.*
   :width: 80 %
   :alt: Animation of agents moving back and forth in a corridor until they reach the end.

.. ATTENTION::
   TODO: Is it safe to have this .gif here when we go to pdf?

Creating the MultiCorridor Simulation
-------------------------------------

The Agents in the Simulation
````````````````````````````
It's helpful to start by thinking about what we want the agents to learn and what
information they will need in order to learn it. In this tutorial, we want to
train agents that can reach the end of a one-dimensional corridor without bumping
into each other. Therefore, agents should be able to move left, move right, and
stay still. In order to move to the end of the corridor without bumping into each
other, they will need to see their own position and if the squares near them are
occupied. Finally, we need to decide how to reward the agents. There are many ways
we can do this, and we should at least capture the following:

* The agent should be rewarded for reaching the end of the corridor.
* The agent should be penalized for bumping into other agents.
* The agent should be penalized for taking too long.

Since all our agents are homogeneous, we can just create them in the Agent Based
Simulation itself, like so:

.. code-block:: python

   from enum import IntEnum

   from gym.spaces import Box, Discrete, MultiBinary
   import numpy as np

   from admiral.envs import Agent, AgentBasedSimulation

   class MultiCorridor(AgentBasedSimulation):

       class Actions(IntEnum): # The tree actions each agent can take
           LEFT = 0
           STAY = 1
           RIGHT = 2

       def __init__(self, end=10, num_agents=5):
           self.end = end
           agents = {}
           for i in range(num_agents):
               agents[f'agent{i}'] = Agent(
                   id=f'agent{i}',
                   action_space=Discrete(3), # Move left, stay still, or move right
                   observation_space={
                       'position': Box(0, self.end-1, (1,), np.int), # Observe your own position
                       'left': MultiBinary(1), # Observe if the left square is occupied
                       'right': MultiBinary(1) # Observe if the right square is occupied
                   }
               )
           self.agents = agents
           
           self.finalize()

Here, notice how the agents `observation_space` is a `dict` rather than a
`gym.space.Dict`. That's okay because our `Agent` class can convert a dict
of gym spaces into a `Dict`. This happens when ``finalize`` is called at the
end of our ``__init__``.


Resetting the Simulation
````````````````````````

At the beginning of each episode, we want the agents to be randomly positioned
throughout the corridor without occupying the same squares. We must give each agent
a position attribute at reset. We will also create a data structure that captures
which agent is in which cell so that we don't have to do a search for nearby agents
but can directly index the space. Finally, we must track the agents' rewards.

.. code-block:: python

   def reset(self, **kwargs):
       location_sample = np.random.choice(self.end-1, len(self.agents), False)
       # Track the squares themselves
       self.corridor = np.empty(self.end, dtype=object)
       # Track the position of the agents
       for i, agent in enumerate(self.agents.values()):
           agent.position = location_sample[i]
           self.corridor[location_sample[i]] = agent
       
       # Track the agents' rewards over multiple steps.
       self.reward = {agent_id: 0 for agent_id in self.agents}


Stepping the Simulation
```````````````````````

Simulation stepping is driven by the agents' actions because there are no other
activities happening in this simulation. Thus, the MultiCorridor Simulation only
concerns itself with processing the agents' actions at each step. For each agent,
we'll capture the following cases:

* An agent attempts to move to a space that is unoccupied.
* An agent attempts to move to a space that is already occupied.
* An agent attempts to move to the right-most space (the end) of the corridor.

.. code-block:: python

   def step(self, action_dict, **kwargs):
       for agent_id, action in action_dict.items():
           agent = self.agents[agent_id]
           if action == self.Actions.LEFT:
               if agent.position != 0 and self.corridor[agent.position-1] is None:
                   # Good move, no extra penalty
                   self.corridor[agent.position] = None
                   agent.position -= 1
                   self.corridor[agent.position] = agent
                   self.reward[agent_id] -= 1 # Entropy penalty
               elif agent.position == 0: # Tried to move left from left-most square
                   # Bad move, only acting agent is involved and should be penalized.
                   self.reward[agent_id] -= 5 # Bad move
               else: # There was another agent to the left of me that I bumped into
                   # Bad move involving two agents. Both are penalized
                   self.reward[agent_id] -= 5 # Penalty for offending agent
                   # Penalty for offended agent 
                   self.reward[self.corridor[agent.position-1].id] -= 2
           elif action == self.Actions.RIGHT:
               if self.corridor[agent.position + 1] is None:
                   # Good move, but is the agent done?
                   self.corridor[agent.position] = None
                   agent.position += 1
                   if agent.position == self.end-1:
                       # Agent has reached the end of the corridor!
                       self.reward[agent_id] += self.end ** 2
                   else:
                   # Good move, no extra penalty
                       self.corridor[agent.position] = agent
                       self.reward[agent_id] -= 1 # Entropy penalty
               else: # There was another agent to the right of me that I bumped into
                   # Bad move involving two agents. Both are penalized
                   self.reward[agent_id] -= 5 # Penalty for offending agent
                   # Penalty for offended agent
                   self.reward[self.corridor[agent.position+1].id] -= 2 
           elif action == self.Actions.STAY:
               self.reward[agent_id] -= 1 # Entropy penalty

.. ATTENTION::
   Our reward schema reveals a training
   dynamic that is not present in single-agent simulations: an agent's reward
   does not entirely depend on its own interaction with the environment but can
   be affected by other agents' interactions with the environment. In this case, agents
   are slightly penalized for being "bumped into" when other agents attempt to move
   onto their square. This is discussed in MARL literature and captured in the way
   we have designed our Simulation Managers. In Admiral, we favor capturing the rewards
   as part of the simulation's state and only "flushing" them once they rewards are
   asked for in ``get_reward``.

.. NOTE::
   We have not needed to consider the order in which the simulation processes actions.
   Our simulation simply provides the capabilities to process *any* agent's action,
   and we can use `Simulation Managers` to impose an order. This shows the flexibility
   of our design. In this tutorial, we use the `TurnBasedManager`, but we can use
   any `SimulationManager`.

Querying Simulation State
`````````````````````````

The trainer needs to see how agents' actions impact the simulation's state. They do
so via getters, which we define below.

.. code-block:: python

   def get_obs(self, agent_id, **kwargs):
       agent_position = self.agents[agent_id].position
       if agent_position == 0 or self.corridor[agent_position-1] is None:
           left = False
       else:
           left = True
       if agent_position == self.end-1 or self.corridor[agent_position+1] is None:
           right = False
       else:
           right = True
       return {
           'position': [agent_position],
           'left': [left],
           'right': [right],
       }
   
   def get_done(self, agent_id, **kwargs):
       return self.agents[agent_id].position == self.end - 1
   
   def get_all_done(self, **kwargs):
       for agent in self.agents.values():
           if agent.position != self.end - 1:
               return False
       return True
   
   def get_reward(self, agent_id, **kwargs):
       agent_reward = self.reward[agent_id]
       self.reward[agent_id] = 0
       return agent_reward
   
   def get_info(self, agent_id, **kwargs):
       return {}

Rendering for Visualization
```````````````````````````
Finally, it's often useful to be able to visualize a simulation as it steps through
an episode. We can do this via the render funciton.

.. code-block:: python

   def render(self, *args, fig=None, **kwargs):
       draw_now = fig is None
       if draw_now:
           from matplotlib import pyplot as plt
           fig = plt.gcf()
   
       fig.clear()
       ax = fig.gca()
       ax.set(xlim=(-0.5, self.end + 0.5), ylim=(-0.5, 0.5))
       ax.set_xticks(np.arange(-0.5, self.end + 0.5, 1.))
       ax.scatter(np.array(
           [agent.position for agent in self.agents.values()]),
           np.zeros(len(self.agents)),
           marker='s', s=200, c='g'
       )
   
       if draw_now:
           plt.plot()
           plt.pause(1e-17)



Training the MultiCorridor Simulation
-------------------------------------

Now that we have created the simulation and agents, we can create a configuration
file for training.

Simulation Setup
````````````````

We'll start by setting up the simulation we have just built.
Then we'll choose a Simulation Manager. Admiral comes with two built-In
managers--TurnBasedManager, where only a single agent take a turn per step, and
AllStepManager, where all non-done agent take a turn per step. For this experiment,
we'll use the TurnBasedManager. Then, we'll wrap the simulation with our `MultiAgentWrapper`,
which enables us to connect with RLlib. Finally, we'll register the simulation
with RLlib.

.. code-block:: python

   # MultiCorridor is the simulation we created above
   from admiral.envs.corridor import MultiCorridor
   from admiral.managers import TurnBasedManager
   # MultiAgentWrapper needed to connect with RLlib
   from admiral.external import MultiAgentWrapper

   # Create an instance of the simulation and register it
   env = MultiAgentWrapper(AllStepManager(MultiCorridor()))
   env_name = "MultiCorridor"
   from ray.tune.registry import register_env
   register_env(env_name, lambda env_config: env)

Policy Setup
````````````

Now we want to create the policies and policy mapping function in our multiagent
experiment. Each agent in our simulation is homogeneous: they all have the same
observation space, action space, and objective function. Thus, we can create a
single policy and map all agents to that policy.

.. code-block:: python

   ref_agent = env.unwrapped.agentsp['agent0']
   policies = {
       'corridor': (None, ref_agent.observation_space, ref_agent.action_space, {})
   }
   def policy_mapping_fn(agent_id):
       return 'corridor'

Experiment Parameters
`````````````````````

Having setup the simulation and policies, we can now bundle all that information
into a parameters dictionary that will be read by Admrial and used to launch RLlib.

.. code-block:: python

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
               # "lr": 0.0001,
               # --- Parallelism ---
               # Number of workers per experiment: int
               "num_workers": 7,
               # Number of environments that each worker starts: int
               "num_envs_per_worker": 1, # This must be 1 because we are not "threadsafe"
           },
       }
   }

Command Line interface
``````````````````````
With the configuration file complete, we can utilize the command line interface
to train our agents. We simply type ``admiral train multi_corridor_example.py``,
where `multi_corridor_example.py` is the name of our configuration file. This will launch
Admiral, which will process the file and launch RLlib according to the
specified parameters. This particular example should take 1-10 minutes to
train, depending on your compute capabilities. You can view the performance
in real time in tensorboard with ``tensorboard --logdir ~/admiral_results``.


Visualizing
-----------
We can visualize the agents' learned behavior with the ``visualize`` command, which
takes as argument the output directory from the training session stored in
``~/admiral_results``. For example, the command

.. code-block::

   admiral visualize ~/admiral_results/MultiCorridor-2020-08-25_09-30/ -n 5 --record

will load the experiment (notice that the directory name is the experiment
name from the configuration file appended with a timestamp) and display an animation
of 5 episodes. The ``--record`` flag will save the animations as `.mp4` videos in
the training directory.
