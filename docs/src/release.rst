.. Abmarl latest releases.

What's New in Abmarl
====================

Bug fixes
* Bugfix command line debug uses -s arg now
* Bugfix inactive agent still block
* Bugfix turn based manager learning agents



Abmarl version 0.2.7 features the new
:ref:`Smart Simulation and Registry <gridworld_smart_sim>`,
which streamlines creating simulations by allowing components to be specified at the
simulation's initialization; a new :ref:`Ammo Agent <gridworld_ammo>` that restricts
how many attacks an agent can issue during the simluation; the ability to
:ref:`barricade a target <gridworld_barricade_placement>` with barriers at the start
of a simulation; and an updated :ref:`Debugger <debugging>` that outputs the log
file by event, so you can see each action and state update in order.

Smart Simulation and Registry
-----------------------------

Previously, changing a component in the simulation required a change to the simulation
definition. For example, changing between the 
:ref:`PositionCenteredEncodingObserver <gridworld_position_centered_observer>` and
the :ref:`AbsoluteEncodingObserver <gridworld_absolute_encoding_observer>` in the
:ref:`Team Battle Simulation <gridworld_tutorial_team_battle>` required users to
manually change the simulation definition or to define multiple simulations that
were exactly the same but had a differet `observer`. The
:ref:`Smart Simulation <gridworld_smart_sim>` streamlines creating
simulations by allowing components to be specified at the simulations' *initialization*,
instead of requiring them to be specified in the simulation *definition*. This avoids
workflow issues where the config file in an output directory is including a different
version of the simulation than what was used in training caused by the user changing
the simulation definition between training runs.

:ref:`States <gridworld_state>`, :ref:`Observers <gridworld_observer>`, and
:ref:`Dones <gridworld_done>` can be given at initialization as the class (e.g.
``TargetDone``). Any :ref:`registered <gridworld_smart_sim>` component can also
be given as the class name (e.g. ``"TargetDone"``). All
:ref:`Built in features <gridworld_built_in_features>` are automatically registered,
and users can :ref:`register <api_gridworld_register>` custom components.

.. NOTE::
   The :ref:`Smart Simulation <gridworld_smart_sim>` cannot currently support
   :ref:`Actors <gridworld_actor>`, so those must still be defined in the simulation
   definition.


Ammo Agents
-----------

:ref:`Ammo Agents <gridworld_ammo>` have limited ammunition that determines how
many attacks they can issue per simualation. The :ref:`Attack Actors <gridworld_attacking>`
interpret the ammunition in conjunction with `simultaneous attacks` to provide
the ability to determine both how many attacks can be issued per step and, with
the addition of Ammo Agents, how many attacks can be issued during the entire simulation.
Agents that have run out of ammo will still be able to chose to attack, but that
attack will be unsuccessful.

Target Barricading
------------------

Similar to the :ref:`MazePlacementState <_gridworld_position_maze_placement>`, Abmarl now
includes the ability to cluster the *barrier* around the target in such a way that
the target is completely enclosed. For example, a target with 8 barriers will provide
a single layer of barricade, 24 barriers two layers, 48 barriers three, and so on
(with some variation if the target starts near an edge or corner). The following
animation shows some example starting states using the
:ref:`TargetBarriersFreePlacementState <gridworld_barricade_placement>`:

.. figure:: /.images/gridworld_blockade_placement.*
   :width: 75 %
   :alt: Animation showing starting states using Target Barrier Free Placement State component.

   Animation showing a target (green) starting at random positions at the beginning
   of each episode. Barriers (gray squares) completely enclose the target. Free
   agents (blue and red) are scattered far from the target.

Debugging by Event
------------------

Abmarl's :ref:`Debugger <debugging>` now outputs log files by agent and by event
to the output directory. The file `Episode_by_agent.txt` organizes the SARS by type
and then by agent, so one can see all the observations made by a specific agent
during the simulation, or all the actions made by another agent during the simulation.
`Episode_by_event.txt`, on the other hand, shows the events in order, starting with
reset and moving through each step.


Miscellaneous
-------------

Interface changes
`````````````````

* :ref:`Attacking Agents <api_gridworld_agent_attack>` `attack_count` has been changed
  to `simultaneous_attacks` to deconflict the concept with the new ammunition feature.
* :ref:`Attack mapping <api_gridworld_actor_attack>` now expects a set of attackable
  encodings instead of a list.
* The SingleGridObserver has been changed to the
  :ref:`PositionCenteredEncodingObserver <api_gridworld_observer_position_centered>`.
* The MultiGridObserver has been changed to the
  :ref:`StackedPositionCenteredEncodingObserver <api_gridworld_observer_position_centered_stacked>`.

Other Features
``````````````

* Abmarl provides a
  `custom box space <https://github.com/LLNL/Abmarl/blob/main/abmarl/tools/gym_utils.py#L6>`_
  that will return true when checking if a single numeric value is *in* a `Box`
  space with dimension 1. That is, Abmarl's `Box` does not distinguish between
  ``[24]`` and ``24``; both are in, say, ``Box(-3, 40, (1,), int)``.
* :ref:`MazePlacementState <api_gridworld_state_position_maze>` can take the target
  agent by object or by id, which is useful in situations where one does not have
  the target object, such as if one is building from an array with an object registry.
* A new :ref:`TargetDestroyedDone <gridworld_done_target_destroyed>`, which is similar to the
  already-existing :ref:`TargetAgentDone <gridworld_done_target_overlap>`, but the
  target must become *inactive* in order for the agent to be considered done.
* Enhanced :ref:`RLlib's wrapper <rllib_external>` for less warnings when training
  with RLlib.

Bug fixes
`````````

* The :ref:`TurnBasedManager <api_turn_based>` no longer expects output from non-learning
  agents, that is, entities in the simulation that are not observing or acting.
* Inactive agents no longer :ref:`block <gridworld_blocking>`.
* The :ref:`Debug command line interface <debugging>` now makes use of the ``-s``
  argument, which specifies simulation horizon (i.e. max steps to take in a single
  run).
