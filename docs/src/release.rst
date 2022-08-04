.. Abmarl latest releases.

What's New in Abmarl
====================

Abmarl version 0.2.4 has some exciting new capabilities. Among these is the ability to
connect Abmarl Simulations with Open Spiel algorithms via the new
:ref:`OpenSpielWrapper <open_spiel_external>`; a one-of-a-kind
:ref:`SuperAgentWrapper <super_agent_wrapper>` for grouping agents; and a prototype
of Abmarl's very own :ref:`Trainer <trainer>` framework for rapid algorithm
development and testing.


OpenSpiel Wrapper
-----------------

The :ref:`OpenSpielWrapper <open_spiel_external>` is an external wrapper alongside
the :ref:`MultiAgentEnvWrapper <rllib_external>`` and the :ref:`GymWrapper <gym_external>`.
The OpenSpiel Wrapper enables the connection between Abmarl's :ref:`SimulationManager <sim-man>`
and OpenSpiel algorithms, increasing Abmarl's ease of use for MARL researchers.

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

Along with this feature, the :ref:`Simulation Managers <sim-man>` now explicity
track the set of done agents.

SuperAgentWrapper
-----------------

Users can setup Abmarl simulations such that multiple agents generate experiences
that are all used to train a single policy. The policy itself is designed for a
single agent's input and output. This method of multiple agents is a way to parallelize
the data generation process and speed up training. It is the method of choice for
collaborative agents.

With the new :ref:`SuperAgentWrapper <super_agent_wrapper>`, users can define groupings
of agents so that a single policy is responsible for digesting all the observations
and generating all the actions for its agents in a single pass.

The :ref:`SuperAgentWrapper <super_agent_wrapper>` can be used with an Abmarl Simulation
and a mapping of *super* agents to *covered* agents, like so:

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

To fully support integration with the RL loop, users can now specify
:ref:`null observations and actions <overview_agent>` for agents.


Null Observations and Actions
-----------------------------

Up until now, any agent that finishes the simulation early will return its final
experience and refrain from further interaction in the simulation. With the introduction
of the :ref:`SuperAgentWrapper <super_agent_wrapper>` and the
:ref:`OpenSpielWrapper <open_spiel_external>`, done agents may still be queried
for their observations and even report actions. In order to keep the training data
*clean*, users can now specify :ref:`null observations and actions <overview_agent>`
for agents, which will be used in these rare cases.


Trainer Prototype
-----------------

The :ref:`Trainer <trainer>` prototype is a first attempt to support Abmarl's
in-house algorithm development. The prototype is built off an on-policy monte-carlo
algorithm and abstracts the data generation process, enabling the user to focus
on developing the training rules. As Abmarl continues to grow, one can expect
more development in the training framework.


Dynamic Order Manager and Simulation
------------------------------------

The new :ref:`DynamicOrderSimulation <api_dynamic_sim>` and
:ref:`DynamicOrderManager <api_dynamic_man>` combo allows users to create
simulations where the simulation itself can determine the next agent(s) to act.


Miscellaneous
-------------

* Checking the ``isinstance`` of an :ref:`Agent <api_agent>` now automatically
captures :ref:`ObservingAgents <api_observing_agent>` and
:ref:`ActingAgents <api_acting_agent>`.
* Example simulations have been centralized in ``abmarl.examples.sim``. These examples
are a store of useful simulations for testing, debugging, understanding RL, etc.
* Updated ray dependency to version 1.12.1. This had the following side-effects:

   * Update Abmarl :ref:`MultiAgentWrapper <rllib_external>` to work with new RLlib interface.
   * Pinned the gym version to be less than 0.22. These version of gym are not
   as clever in decided whether a point is in a space, so `Box` spaces must now
   explicitly output a list or array, even if there is only a single element.
   * Users may want to make use of the new ``disable_env_checking`` flag available
   in RLlib's configuration.

* Done agents are not removed from the Grid, so agents that cannot overlap are
needlessly restricted. Grid overlapping logic now checks if an agent is done.
