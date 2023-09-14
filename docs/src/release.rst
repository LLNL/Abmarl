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





Absolute Grid Observer
----------------------

The :ref:`Single and Multi Grid Observers <gridworld_single_observer>` provide
observations of the the grid centered on the observing agent, a view of the grid
"from the agent's perspective". Abmarl's :ref:`Grid World Simulation Framework <gridworld>`
now contains the :ref:`Absolute Grid Observer <gridworld_absolute_grid_observer>`,
which produces observations of the grid "from the `grid's` perspective". The observation
size matches the size of the grid, and the agent sees itself moving around the
grid instead of seeing all the other agents positioned relative to itself.

Here we show the following state observations for the bottom-left red agent with
a ``view_range`` of 2 via the :ref:`Single Grid Observer <gridworld_single_observer>`
and the new :ref:`Absolute Grid Observer <gridworld_absolute_grid_observer>`. The
Single Grid Observation is sized by the agent's view range, the observing agent
is in the very center, and all other cells are shown by their relative positions,
including out of bounds cells. The Absolute Grid Observation is sized by the grid,
all agents are shown in their actual grid positions, there are no out of bounds
cells, and any cell that the agent cannot see is masked with a -2.

.. figure:: /.images/absolute_vs_position_obs.png
   :width: 75 %
   :alt: State for comparing the differences between single and absolute grid observer.

   Comparing observations for the bottom-left red agent with a ``view_range`` of 2.
   The green agent has an encoding of 1, the gray agents 2, and the red agents 3.
   
.. code-block::

   # Single Grid Observer, observing agent is shown here as *3
   [ 0,  2,  2,  0,  2],
   [ 0,  2,  0,  0,  0],
   [ 0,  0, *3,  3,  0],
   [ 0,  0,  0,  0,  0],
   [-1, -1, -1, -1, -1],

   # Absolute Grid Observer, observing agent is shown as -1
   [-2, -2, -2, -2, -2, -2, -2],
   [-2, -2, -2, -2, -2, -2, -2],
   [-2, -2, -2, -2, -2, -2, -2],
   [ 0,  2,  2,  0,  2, -2, -2],
   [ 0,  2,  0,  0,  0, -2, -2],
   [ 0,  0, -1,  3,  0, -2, -2],
   [ 0,  0,  0,  0,  0, -2, -2]


Maze Placement State
--------------------

The :ref:`Position State <gridworld_position>` supports placing agents in the the
grid either (1) according to their initial positions or (2) by randomly selecting
an available cell. The new :ref:`Maze Placement State <gridworld_position_maze_placement>`
supports more structure in initially placing agents. It starts by partitioning
the grid into two types of cells, `free` or `barrier`, according to a maze that
is generated starting at some `target agent's` position. Agents with `free encodings`
and `barrier encodings` are then randomly placed in `free` cells and `barrier` cells,
respectively. The Maze Placement State component can be configured such that it
clusters `barrier` agents near the target and scatters `free` agents away from
the target. The clustering is such that all paths to the target are not blocked.
In this way, the grid can be randomized at the start of each episode, while still
maintaining some desired structure.

.. figure:: /.images/gridworld_maze_placement.*
   :width: 75 %
   :alt: Animation showing starting states using Maze Placement State component.

   Animation showing a target (green) starting at random positions at the beginning
   of each episode. Barriers (gray squares) are clustered near the target without
   blocking all paths to it. Free agents (blue) are scattered far from the target.


Building a Gridworld Simulation
-------------------------------

Abmarl's :ref:`Gridworld Simulation Framework <gridworld>` now supports
:ref:`building the simulation <gridworld_building>` in these ways:

   #. Building the simulation by specifying the rows, columns, and agents;
   #. Building the simulation from an existing :ref:`grid <gridworld_grid>`;
   #. Building the simulation from an array and an object registry; and
   #. Building the simulation from a file and an object registry.

Additionally, when building the simulation from a grid, array, or file, you can
specify additional agents to build that are not in those inputs. The builder will
combine the content from the grid, array, or file with the extra agents.


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
