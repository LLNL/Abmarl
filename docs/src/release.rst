.. Abmarl latest releases.

What's New in Abmarl
====================

Top features
* Smart Agent Based Simulation
* Debug outputs play-by-play
* State component for barricading a target with barriers
* Agents with ammo and attackers work with ammo

Interface changes
* Change name of attack count to simultaneous attacks
* Attack mapping expects a set
* New names for Single and Multi Grid observers

Other features
* Maze placement state can take target by id
* Built in support for target destroyed done
* Custom Box space that returns true for single numeric in space. GIVE EXAMPLE.
* Smarter prechecker in env wrapper for better integration with rllib

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
   The :ref:`Smart Simulation <gridworld_smart_sim>`` cannot currently support
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

Similar to the :ref:`MazePlacementState <gridworld_maze_placement>`, Abmarl now
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

* New built-in :ref:`Target agent component <gridworld_done_built_in>` supports
  agents having a target agent with which they must overlap.
* New :ref:`Cross Move Actor <gridworld_movement_cross>` allows the agents to move
  up, down, left, right, or stay in place.
* The :ref:`All Step Manager <api_all_step>` supports randomized ordering in the
  action dictionary.
* The :ref:`Position State <gridworld_position>` component supports ignoring the
  overlapping options during random placement. This results in agents being placed
  on unique cells.
* Abmarl's visualize component now supports the ``--record-only`` flag, which will
  save animations without displaying them on screen, useful for when running headless
  or processing in batch.
* Bugfix with the :ref:`Super Agent Wrapper <super_agent_wrapper>` enables training
  with rllib 2.0.
* Abmarl now supports Python 3.9 and 3.10.
* Abmarl now supports gym 0.23.1.
