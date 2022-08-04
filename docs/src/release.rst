.. Abmarl latest releases.

What's New in Version 0.2.4
===========================

Abmarl version 0.2.4 has some exciting new capabilities. Among these is the ability to
connect Abmarl Simulations with Open Spiel algorithms via the new OpenSpiel Wrapper;
a one-of-a-kind SuperAgentWrapper for defining groupings of agents; and a prototype
of Abmarl's very own Trainer framework for rapid algorithm development and testing.


OpenSpiel Wrapper
-----------------

The OpenSpiel Wrapper is an external wrapper alongside the MultiAgentEnvWrapper
and the GymWrapper. The OpenSpiel Wrapper enables the connection between Abmarl's
SimulationManager and OpenSpiel algorithms, increasing Abmarl's ease of use for
MARL researchers.

.. code-block:: python

   sim = OpenSpielWrapper(
       AllStepManager(MultiCorridor())
   ) # sim is ready to use with an open-spiel algorithm.
   time_step = sim.reset()
   for _ in range(20):
       agents_output = [trainer.step(time_step) for trainer in trainers.values()]
       action_list = [agent_output.action for agent_output in agents_output]
       assert len(action_list) == 5
       time_step = sim.step(action_list)
       if time_step.last():
           for trainer in trainers.values():
               trainer.step(time_step)
               break
   for trainer in trainers.values():
       trainer.step(time_step)

Along with this feature, the Simulation Managers now explicity track the set of
done agents.

SuperAgentWrapper
-----------------

Users can setup Abmarl simulations such that multiple agents generate experiences
that are all used to train a single policy. The policy itself is designed for a
single agent's input and output. This method of multiple agents is a way to parallelize
the data generation process and speed up training. It is the method of choice for
collaborative agents.

With the new SuperAgentWrapper, users can define groupings of agents so that a single
policy is responsible for digesting all the observations and generating all the
actions for its agents in a single pass.

The SuperAgentWrapper can be used with an Abmarl Simulation and a mapping of *super*
agents to *covered* agents, like so:

.. code-block:: python

   AllStepManager(
       SuperAgentWrapper(
           TeamBattleSim.build_sim(
               8, 8,
               agents=agents,
               overlapping=overlap_map,
               attack_mapping=attack_map
           ),
           super_agent_mapping = {
               'red': [agent.id for agent in agents.values() if agent.encoding == 1],
               'blue': [agent.id for agent in agents.values() if agent.encoding == 2],
               'green': [agent.id for agent in agents.values() if agent.encoding == 3],
               'gray': [agent.id for agent in agents.values() if agent.encoding == 4],
           }
       )
   )

To fully support integration with the RL loop, users can now specify null observations
and actions for agents.


Null Observations and Actions
-----------------------------

Up until now, any agent that finishes the simulation early will return its final
experience and refrain from further interaction in the simulation. With the introduction
of the SuperAgentWrapper and the OpenSpielWrapper, done agents may still be queried
for their observations and even report actions. In order to keep the training data
"clean", users can now specify null observations and actions for agents, which
will be used in these rare cases.


Trainer Prototype
-----------------

The Trainer prototype is a first attempt to support Abmarl's in-house algorithm development.
The prototype is built off an on-policy monte-carlo algorithm and abstracts the
data generation process, leaving the user to focus on developing the training rules.
As Abmarl continues to grow, one can expect more development in the training framework.

# TODO:
Added Prototype for Trainer framework.
Limits:
* based on Monte Carlo algorithms.
* on policy
* single agent per policy.
Expect more development here.

* debug through random trainer. Show example.


Dynamic Order Manager and Simulation
------------------------------------

The new DynamicOrderSimulation and DynamicOrderManager combo allows users to create
simulations where the simulation itself can determine the next agent(s) to act.

TODO:
 nad decided by the Simulation.
The order of the agents is dynamically decided by the simulation as it runs.
The simulation must be a DynamicOrderSimulation. The agents reported at reset
and step are those given in the sim's next_agent property.

Assert that the incoming action does not come from an agent who is recorded
as done. Step the simulation forward and return the observation, reward,
done, and info of the next agent. The simulation is responsible to ensure
that there is at least one next_agent that did not finish in this turn,
unless it is the last tur

An AgentBasedSimulation where the simulation chooses the agents' turns dynamically.

```
@property
def next_agent(self):
    """
    The next agent(s) in the game.
    """
    return self._next_agent

@next_agent.setter
def next_agent(self, value):
    assert isinstance(value, (Container, str)), \
        "The next agent must be a single string or a Container of strings."
    if type(value) == str:
        value = [value]
    for agent_id in value:
        assert agent_id in self.agents, \
            "Every next agent must be an agent in the simulation."
    self._next_agent = value


sim = SequentiallyFinishingSim()
sim.next_agent = 'agent0'
assert sim.next_agent == ['agent0']
sim.next_agent = ['agent1', 'agent2']
assert sim.next_agent == ['agent1', 'agent2']
sim.next_agent = ('agent3',)
assert sim.next_agent == ('agent3',)
sim.next_agent = set(('agent0', 'agent1'))
assert sim.next_agent == set(('agent0', 'agent1'))
```


Miscellaneous
-------------

* isinstance for Agent object: now Agent(ObservingAgent, ActingAgent) really means something
* cleaner examples and tests. Examples found in abmarl.examples. Useful simulations
for testing, debugging, understanding, etc.
* Updated ray dependency. Currently  version 1.12.1. Changes in the MultiAgentEnvWrapper
to work with new RLlib interface.
    - Pinned gym version
    - Disable env checking
    - Gym spaces are stricter: [0] vs 0.
* Grid overlapping fix for inactive agents





.. _reference:

Referencce
``````````

First, we have :ref:`Agents <api_agent>`. An agent is an object with an observation and
action space. Many practitioners may be accustomed to gym.Env's interface, which
defines the observation and action space for the *simulation*. However, in heterogeneous
multiagent settings, each *agent* can have different spaces; thus we assign these
spaces to the agents and not the simulation.

An agent can be created like so:

.. code-block:: python

   from gym.spaces import Discrete, Box
   from abmarl.sim import Agent
   agent = Agent(
       id='agent0',
       observation_space=Box(-1, 1, (2,)),
       action_space=Discrete(3)
   )

At this level, the Agent is basically a dataclass. We have left it open for our
users to extend its features as they see fit.

.. WARNING::
   Implementations of AgentBasedSimulation should call ``finalize`` at the
   end of its ``__init__``. Finalize ensures that all agents are configured and
   ready to be used for training.

.. NOTE::
   Instead of treating agents as dataclasses, we could have included the relevant
   information in the Agent Based Simulation with various dictionaries. For example,
   we could have ``action_spaces`` and ``observation_spaces`` that
   maps agents' ids to their action spaces and observation spaces, respectively.
   In Abmarl, we favor the dataclass approach and use it throughout the package
   and documentation.

The Agent Based Simulation interface does not specify an ordering for agents' interactions
with the simulation. This is left open to give our users maximal flexibility. However,
in order to interace with RLlib's learning library, we provide a :ref:`Simulation Manager <api_sim>`
which specifies the output from ``reset`` and ``step`` as RLlib expects it. Specifically,

1. Agents that appear in the output dictionary will provide actions at the next step.
2. Agents that are done on this step will not provide actions on the next step.

Simluation Managers "wrap" simulations, and they can be used like so:

.. code-block:: python

   from abmarl.managers import AllStepManager
   from abmarl.sim import AgentBasedSimulation, Agent
   class MySim(AgentBasedSimulation):
       ... # Define some simulation

   # Instatiate the simulation
   sim = MySim(agents=...)
   # Wrap the simulation with the simulation manager
   sim = AllStepManager(sim)
   # Get the observations for all agents
   obs = sim.reset()
   # Get simulation state for all non-done agents, regardless of which agents
   # actually contribute an action.
   obs, rewards, dones, infos = sim.step({'agent0': 4, 'agent2': [-1, 1]})

.. WARNING::
   The :ref:`Dynamic Order Manager <api_dynamic_man>` must be used with a
   :ref:`Dynamic Order Simulation <api_dynamic_sim>`. This allows the simulation
   to dynamically choose the agents' turns, but it also requires the simulation
   to pay attention to the interface rules. For example, a Dynamic Order Simulation
   must ensure that at every step there is at least one reported agent who is not done,
   unless it is the last turn.


The experiment parameters also contains information that will be passed directly
to RLlib via the `ray_tune` parameter. See RLlib's documentation for a
`list of common configuration parameters <https://docs.ray.io/en/releases-1.2.0/rllib-training.html#common-parameters>`_.
