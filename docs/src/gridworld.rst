.. Abmarl gridworld documentation

GridWorld Simulation Framework
==============================

Abmarl provides a GridWorld Simulation Framework for setting up grid-based
Agent Based Simulations, which can be connected to Reinforcement Learning algorithms
through Abmarl's :ref:`AgentBasedSimulation <abs>` interface. The GridWorld
Simulation Framework is a `grey box`: we assume users have working knowledge of 
Python and object-oriented programming. Using the
:ref:`built in features <gridworld_built_in_features>` requires minimal knowledge,
but extending them and creating new features requires more knowledge.
In addition to the design documentation below, see the :ref:`GridWorld tutorials <>`
for in-depth examples on using and extending the GridWorld Simulation Framework.


Framework Design
----------------

.. figure:: .images/gridworld_framework.png
   :width: 100 %
   :alt: Gridworld Simulation Framework

   Abmarl's GridWorld Simulation Framework. A simulation has a Grid, a dictionary
   of agents, and various components that manage the various features of the simulation.

The GridWorld Simulation Framework utilizes a modular design that allows users
to create new features and plug them in as components of the simulation. Every component
inherits from the :ref:`GridWorldBaseComponent <>` class and has a reference to
a :ref:`Grid <gridworld_grid>` and a dictionary of :ref:`Agents <gridworld_agent>`.
These components make up a :ref:`GridWorldSimulation <>`, which extends the
:ref:`AgentBasedSimulation <abs>` interface. For example, a simulation might look
something like this:

TODO: Check that all this code works!

.. code-block:: python

   from abmarl.sim.gridworld.base import GridWorldSimulation
   from abmarl.sim.gridworld.state import PositionState
   from abmarl.sim.gridworld.actor import MoveActor
   from abmarl.sim.gridworld.observer import SingleGridObserver
   
   class MyGridSim(GridWorldSimulation):
       def __init__(self, **kwargs):
           self.agents = kwargs['agents']
           self.position_state = PositionState(**kwargs)
           self.move_actor = MoveActor(**kwargs)
           self.observer = SingleGridObserver(**kwargs)

       def reset(self, **kwargs):
           self.position_state.reset(**kwargs)
       
       def step(self, action_dict):
           for agent_id, action in action_dict.items():
               self.move_actor.process_action(self.agents[agent_id], action)
    
       def get_obs(self, agent_id, **kwargs):
           return self.observer.get_obs(self.agents[agent_id])


.. _gridworld_agent:

Agent
`````

Every entity in the simulation is a :ref:`GridWorldAgent <>`
(e.g. walls, foragers, resources, fighters, etc.). GridWorldAgents are :ref:`PrincipleAgents <api_agent>` with specific parameters
that work with their respective components. In particular, agents must be given
an `encoding`, which is an integer that correlates to the type of agent and simplifies
the logic for many components of the framework. GridWorldAgents can also be configured
with an `initial position`, the ability to `block` other agents' abilities, and visualization
parameters such as `shape` and `color`.

Following the dataclass model, additional agent classes can be defined that allow
them to work with various components. For example, :ref:`GridObservingAgents <gridworld_single_observer>` can work with
:ref:`Observers <gridworld_single_observer>`, and :ref:`MovingAgents <gridworld_movement>` can work with the :ref:`MoveActor <gridworld_movement>`. Any new agent class should
inhert from :ref:`GridWorldAgent <>` and possibly from :ref:`ActingAgent <api_acting_agent>` or :ref:`ObservingAgent <api_observing_agent>` as needed.
For example, one can define a new type of agent like so:

.. code-block:: python

   class CommunicatingAgent(GridWorldAgent, ActingAgent):
       def __init__(self, broadcast_range=None, **kwargs):
           super().__init__(**kwargs)
           self.broadcast_range = broadcast_range
           ...

.. WARNING::
   Agents should follow the dataclass model, meaning that they should only be given
   parameters. All functionality should be written in the simulation components.


.. _gridworld_grid:

Grid
````
The :ref:`Grid <>` stores :ref:`Agents <gridworld_agent>` in a two-dimensional numpy array. The Grid is configured
to be a certain size (rows and columns) and to allow types of Agents to overlap
(occupy the same cell). For example, you may want a ForagingAgent to be able to overlap
with a ResourceAgent but not a WallAgent. The `overlapping` parameter
is a dictionary that maps the Agent's `encoding` to a list of other Agents' `encodings`
with which it can overlap. For example,

.. code-block:: python

   overlapping = {
       1: [2],
       2: [1, 3],
       3: [2, 3]
   }
   grid = Grid(5, 6, overlapping=overlapping)

means that agents whose `encoding` is 1 can overlap with other agents whose `encoding`
is 2; agents whose `encoding` is 2 can overlap with other agents whose `encoding` is
1 or 3; and agents whose `encoding` is 3 can overlap with other agents whose `encoding`
is 2 or 3.

.. WARNING::
   To avoid undefined behavior, the `overlapping` should be symmetric, so that if
   2 can overlap with 3, then 3 can also overlap with 2.

.. NOTE::
   If `overlapping` is not specified, then no agents will be able to occupy the same
   cell in the Grid.

Interaction between simulation components and the :ref:`Grid <>` is
`data open`, which means that we allow components to access the internals of the
Grid. Although this is possible and sometimes necessary, the Grid also provides
an interface for safer interactions with components. Components can ``query`` the
Grid to see if an agent can be placed at a specific position. Components can ``place``
agents at a specific position in the Grid, which will succeed if that cell is available
to the agent as per the `overlapping` configuration. And Components can ``remove``
agents from specific positions in the Grid. 


State
`````

:ref:`State Components <>` manage the state of the simulation alongside the :ref:`Grid <gridworld_grid>`.
At the bare minimum, each State resets the part of the simulation that it manages
at the the start of each episode.

Actor
`````

:ref:`Actor Components <>` are responsible for processing agent actions and producing changes
to the state of the simulation. Actors assign supported agents with an appropriate
action space and process agents' actions based on the Actor's key. The result of
the action is a change in the simulation's state, and Actors should return that
change in a reasonable form. For example, the :ref:`MoveActor <gridworld_movement>` appends :ref:`MovingAgents' <gridworld_movement>` action
spaces with a 'move' channel and looks for the 'move' key in the agent's incoming
action. After a move is processed, the MoveActor returns how much the agent actually
moved.

Observer
````````

:ref:`Observer Components <>` are responsible for creating an
agent's observation of the state of the simulation. Observers assign supported agents
with an appropriate observation space and generate observations based on the
Observer's key. For example, the :ref:`SingleGridObserver <gridworld_single_observer>` generates an observation and
stores it in the 'grid' channel of the :ref:`ObservingAgent's <gridworld_single_observer>` observation.

Done
````

:ref:`Done Components <>` manage the "done state" of each agent and of the simulation as a
whole. Agents that are reported as done will cease sending actions to the simulation, 
and the episode will end when all the agents are done or when the simulation is done.


.. _gridworld_built_in_features:

Built-in Features
-----------------

TODO: Add graphics showcasing the features

Below is a list of some features that are available to use out of the box. Rememeber,
you can :ref:`create your own features <>` in the GridWorld Simulation Framework
and use many combinations of components together to make up a simulation.


Position
````````
:ref:`Agents <gridworld_agent>` have `positions` in the :ref:`Grid <gridworld_grid>` that are managed by the
:ref:`PositionState <>`. Agents
can be configured with an `initial position`, which is where they will start at the
beginning of each episode. If they are not given an `initial position`, then they
will start at a random cell in the grid. Agents can overlap according to the
:ref:`Grid's <gridworld_grid>` `overlapping` configuration. For example, consider the following setup:

.. code-block:: python

   agent0 = GridWorldAgent(
       id='agent0',
       encoding=1,
       initial_position=np.array([2, 4])
   )
   agent1 = GridWorldAgent(
       id='agent1',
       encoding=1
   )
   position_state = PositionState(
       agents={'agent0': agent0, 'agent1': agent1},
       grid=Grid(4, 5)
   )
   position_state.reset()

`agent0` is configured with an `initial position` and `agent1` and is not. At the
start of each episode, `agent0` will be placed at (2, 4) and `agent1` will be placed
anywhere in the grid (except for (2,4) because they cannot overlap).


.. _gridworld_movement:

Movement
````````

:ref:`MovingAgents <>` can move around the :ref:`Grid <gridworld_grid>` in conjunction with the :ref:`MoveActor <>`. MovingAgents
require a `move range` parameter, indicating how many spaces away they can move
in a single step. Agents cannot move out of bounds and can only move to the same
cell as another agent if they are allowed to overlap. For example, in this setup

.. code-block:: python

   agents = {
       'agent0': MovingAgent(
           id='agent0', encoding=1, move_range=1, initial_position=np.array([2, 2])
       ),
       'agent1': MovingAgent(
           id='agent'1, encoding=1, move_range=2, initial_position=np.array([0, 2])
       )
   }
   grid = Grid(5, 5, overlapping={1: [1]})
   position_state = PositionState(agents=agents, grid=grid)
   move_actor = MoveActor(agents=agents, grid=grid)

   position_state.reset()
   move_actor.process_move(agents['agent0'], {'move': np.array([0, 1])})
   move_actor.process_move(agents['agent1'], {'move': np.array([2, 1])})

`agent0` starts at position (2, 2) and can move up to one cell away. `agent1`
starts at (0, 2) and can move up to two cells away. The two agents can overlap
each other, so when the move actor processes their actions, both agents will be
at position (2, 3).


.. _gridworld_single_observer:

Single Grid Observer
````````````````````

:ref:`GridObservingAgents <>` can observe the state of the :ref:`Grid <gridworld_grid>` around them, namely which
other agents are nearby, via the :ref:`SingleGridObserver <>`. The SingleGridObserver generates
a two-dimensional array sized by the agent's `view range` with the observing
agent located at the center of the array. All other agents within the `view range` will
appear in the observation, shown as their `encoding`. For example, the following setup

.. code-block:: python

   agents = {
       'agent0': GridObservingAgent(id='agent0', encoding=1, initial_position=np.array([2, 2]), view_range=3),
       'agent1': GridWorldAgent(id='agent1', encoding=2, initial_position=np.array([0, 1])),
       'agent2': GridWorldAgent(id='agent2', encoding=3, initial_position=np.array([1, 0])),
       'agent3': GridWorldAgent(id='agent3', encoding=4, initial_position=np.array([4, 4])),
       'agent4': GridWorldAgent(id='agent4', encoding=5, initial_position=np.array([4, 4]))
       'agent5': GridWorldAgent(id='agent5', encoding=6, initial_position=np.array([5, 5]))
   }
   grid = Grid(6, 6, overlapping={4: [5], 5: [4]})
   position_state = PositionState(agents=agents, grid=grid)
   observer = SingleGridObserver(agents=agents, grid=grid)

   position_state.reset()
   observer.get_obs(agents['agent0'])

will output an observation for `agent0` like so:

.. code-block::

   [-1, -1, -1, -1, -1, -1, -1],
   [-1,  0,  2,  0,  0,  0,  0],
   [-1,  3,  0,  0,  0,  0,  0],
   [-1,  0,  0,  1,  0,  0,  0],
   [-1,  0,  0,  0,  0,  0,  0],
   [-1,  0,  0,  0,  0, 4*,  0],
   [-1,  0,  0,  0,  0,  0,  6]

Since `view range` is the number of cells away that can be observed, the observation size is
``(2 * view_range + 1) by (2 * view_range + 1)``. `agent0` is centered in the middle
of this array, shown by its `encoding`: 1. All other agents appear in the observation
relative to its position and shown by their `encodings`. The agent observes some out
of bounds cells, which appear as -1s. `agent3` and `agent4` occupy the same cell,
and the :ref:`SingleGridObserver <>` will randomly select between their `encodings`
for the observation.


.. _gridworld_blocking:

Blocking
~~~~~~~~

Agents can block other agents' abilities and characteristics, such as blocking
them from view, which masks out parts of the observation. For example,
if `agent4` is configured with ``blocking=True``, then the observation would like
like this:

.. code-block::

   [-1, -1, -1, -1, -1, -1, -1],
   [-1,  0,  2,  0,  0,  0,  0],
   [-1,  3,  0,  0,  0,  0,  0],
   [-1,  0,  0,  1,  0,  0,  0],
   [-1,  0,  0,  0,  0,  0,  0],
   [-1,  0,  0,  0,  0, 4*,  0],
   [-1,  0,  0,  0,  0,  0, -2]

The -2 indicates that the cell is masked, and the choice of displaying `agent3`
over `agent4` is still a random choice. Which cells get masked by blocking
agents is determined by drawing two lines
from the center of the observing agent's cell to the corners of the blocking agent's
cell. Any cell whose center falls between those two lines will be masked, as shown BELOW.

TODO: Make a figure showing how view_blocking works.


Multi Grid Observer
```````````````````

Similar to the :ref:`SingleGridObserver <>`, the :ref:`MultiGridObserver <>` displays a separate array
for every `encoding`. Each array shows the relative positions of the agents and the
number of those agents that occupy each cell. Out of bounds indicators (-1) and
masked cells (-2) are present in every grid. For example, the above setup would
show an observation like so:

.. code-block::

   # Encoding 1
   [-1, -1, -1, -1, -1, -1, -1],
   [-1,  0,  0,  0,  0,  0,  0],
   [-1,  0,  0,  0,  0,  0,  0],
   [-1,  0,  0,  1,  0,  0,  0],
   [-1,  0,  0,  0,  0,  0,  0],
   [-1,  0,  0,  0,  0,  0,  0],
   [-1,  0,  0,  0,  0,  0, -2]

   # Encoding 2
   [-1, -1, -1, -1, -1, -1, -1],
   [-1,  0,  1,  0,  0,  0,  0],
   [-1,  0,  0,  0,  0,  0,  0],
   [-1,  0,  0,  0,  0,  0,  0],
   [-1,  0,  0,  0,  0,  0,  0],
   [-1,  0,  0,  0,  0,  0,  0],
   [-1,  0,  0,  0,  0,  0, -2]
   ...

:ref:`MultiGridObserver <>` may be preferable to :ref:`SingleGridObserver <>` in simulations where
there are many overlapping agents.


Health
``````

:ref:`HealthAgents <>` track their `health` throughout the simulation. `Health` is always bounded
between 0 and 1. Agents whose `health` falls to 0 are marked as `inactive`. They can be given an
`initial health`, which they start with at the beginning of the episode. Otherwise,
their `health` will be a random number between 0 and 1, as managed by the :ref:`HealthState <>`.
Consider the following setup:

.. code-block:: python

   agent0 = HealthAgent(id='agent0', encoding=1)
   grid = Grid(3, 3)
   agents = {'agent0': agent0}
   health_state = HealthState(agents=agents, grid=grid)
   health_state.reset()

`agent0` will be assigned a random `health` value between 0 and 1.


Attacking
`````````

`Health` becomes more interesting when we let agents attack one another. :ref:`AttackingAgents <>`
work in conjunction with the :ref:`AttackActor <>`. They have an `attack range`, which dictates
the range of their attack; an `attack accuracy`, which dictates the chances of the
attack being successful; and an `attack strength`, which dictates how much `health`
is depleted from the attacked agent. An agent's choice to attack is a boolean--either
attack or don't attack--and then the AttackActor determines the successfulness
based on the state of the simulation and the attributes of the AttackingAgent. The AttackActor
requires an `attack mapping` dictionary which determines which `encodings` can attack
other `encodings`, similar to the `overlapping` parameter for the :ref:`Grid <gridworld_grid>`. Consider the
following setup:

.. code-block:: python

   agents = {
       'agent0': AttackingAgent(
           id='agent0',
           encoding=1,
           initial_position=np.array([0, 0]),
           attack_range=1,
           attack_strength=1,
           attack_accuracy=1
       ),
       'agent1': HealthAgent(id='agent1', encoding=2, initial_position=np.array([1, 0])),
       'agent2': HealthAgent(id='agent2', encoding=3, initial_position=np.array([0, 1]))
   }
   grid = Grid(2, 2)
   position_state = PositionState(agents=agents, grid=grid)
   health_state = HealthState(agents=agents, grid=grid)
   attack_actor = AttackActor(agents=agents, grid=grid, attack_mapping={1: [2]})

   position_state.reset()
   health_state.reset()
   attack_actor.process_action(agents['agent0'], {'attack': True})
   attack_actor.process_action(agents['agent0'], {'attack': True})

Here, `agent0` attempts to make two attack actions. The first one is successful
because `agent1` is within its `attack range` and is attackable according to the
`attack mapping`. `agent1`'s `health` will be depleted by 1, and as a result its `health`
will fall to 0 and it will be marked as `inactive`. The second attack fails because,
although `agent2` is within range, it is not a type that `agent0` can attack.

.. NOTE::

   Attacks can be blocked by :ref:`blocking <gridworld_blocking>` agents. If an attackable agent is
   masked from an attacking agent, then it cannot be attacked by that agent. The
   masking is determined the same way as the view blocking above.
