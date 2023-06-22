.. Abmarl latest releases.

What's New in Abmarl
====================

Abmarl version 0.2.6 features the new
:ref:`Absolute Grid Observer <gridworld_absolute_grid_observer>`, which produces
"top-down" observations of the grid from the grid's perspective; the
:ref:`Maze Placement State <gridworld_position_maze_placement>` component for structuring
the initial placement of agents within a grid while allowing for variation in each
episode; and enhanced support for :ref:`buildling gridworld simulations <gridworld_building>`.


Absolute Grid Observer
----------------------

The :ref:`Single and Multi Grid Observers <gridworld_single_observer>` provide
observations of the the grid centered on the observing agent, a view of the grid
"from the agent's perspective". Abmarl's :ref:`Grid World Simulation Framework <gridworld>`
now contains the :ref:`Absolute Grid Observer <gridworld_absolute_grid_observer>`,
which produces observations of the grid "from the grid's perspective". The observation
size matches the size of the grid, and the agent sees itself moving around the
grid instead of seeing all the other agents positioned relative to itself.

Here we show the following state observation for the bottom-left red agent with
a ``view_range`` of 2 via the :ref:`Single Grid Observer <gridworld_single_observer>`
and the new :ref:`Absolute Grid Observer <gridworld_absolute_grid_observer>`. The
Single Grid Observation is sized by the agent's view range, the observing agent
is in the very center, and all other cells are shown in their respective positions,
including out of bounds cells. The Absolute Grid Observation is sized by the grid,
all agents are shown in their actual grid positions, there are no out of bounds
cells, and any cell that the agent cannot see is masked with a -2.

.. figure:: /.images/absolute_vs_position_obs.png
   :width: 75 %
   :alt: State for comparing the differences between single and absolute grid observer.

   Comparing observations for the bottom-left red agent with a ``view_range`` of 2.
   
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
grid either (1) according to their initial positions or (2) randomly selecting
an available cell. The new :ref:`Maze Placement State <gridworld_position_maze_placement>`
supports more structure in initially placing agents. It starts by partitioning
the grid into two types of cells, `free` or `barrier`, according to a maze that
is generated starting at some `target agent's` position. Agents with `free encodings`
and `barrier encodigns` are then randomly placed in `free` cells and `barrier` cells,
respectively. The Maze Placement State component can be configured such that it
clusters `barrier` agents near the target and scatters `free` agents away from
the target. The clustering is such that all paths to the target are not blocked.
In this way, the grid can be randomized at the start of each episode, while still
maintaining some desired structure.

.. figure:: /.images/gridworld_maze_placement.*
   :width: 75 %
   :alt: Animation showing starting states using Maze Placement State component.

   Animation showing a target (green) starting at random positions at the beginning
   of each episode. Barriers (grey squares) are clustered near the target without
   blocking all paths to it. Free agents (blue) are scattered far from the target.


Building a Gridworld Simulation
-------------------------------

Abmarl's :ref:`Gridworld Simulation Framework <gridworld>` now supports
:ref:`building the simulation <gridworld_building>` in these ways:

   #. Building the simulation by specifying the rows, columns, and agents,
   #. Building the simulation from an existing :ref:`grid <gridworld_grid>`,
   #. Building the simulation from an array and an object registry, and
   #. Building the simulation from a file and an object registry.

Additionally, when building the simulation from a grid, array, or file, you can
specify additional agents to build that are not in those inputs. The builder will
combine the content from the grid, array, or file with the extra agents.


Miscellaneous
-------------

* New built-in :ref:`Target agent component <gridworld_done_built_in>` supports
  agents having a target agent with which they must overlap.
* New :ref:`Cross Move Actors <gridworld_movement_cross>` allows the agents to move
  up, down, left, right, or stay in place.
* The :ref:`All Step Manager <api_all_step>` supports randomized ordering in the
  action dictionary.
* The :ref:`Position State <gridworld_position>` component supports ignoring the
  overlapping options during randomly placement. This results in agents being placed
  on unique cells.
* Abmarl's visualize component now supports ``--record-only``, which will save animations
  without displaying them on screen, useful for when running headless or processing
  in batch.
* Bugfix with the :ref:`Super Agent Wrapper <super_agent_wrapper>` enables training
  with rllib 2.0.
* Abmarl now supports Python 3.9 and 3.10.
* Abmarl now supports gym 0.23.1.
