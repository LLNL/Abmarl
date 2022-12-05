.. Abmarl gridworld documentation

.. _gridworld:

GridWorld Simulation Framework
==============================

Abmarl provides a GridWorld Simulation Framework for setting up grid-based
Agent Based Simulations, which can be connected to Reinforcement Learning algorithms
through Abmarl's :ref:`AgentBasedSimulation <abs>` interface. The GridWorld
Simulation Framework is a `grey box`: we assume users have working knowledge of 
Python and object-oriented programming. Using the
:ref:`built in features <gridworld_built_in_features>` requires minimal knowledge,
but extending them and creating new features requires more knowledge.
In addition to the design documentation below, see the :ref:`GridWorld tutorials <tutorials_gridworld>`
for in-depth examples on using and extending the GridWorld Simulation Framework.


Framework Design
----------------

.. figure:: .images/gridworld_framework.png
   :width: 100 %
   :alt: Gridworld Simulation Framework

   Abmarl's GridWorld Simulation Framework. A simulation has a Grid, a dictionary
   of agents, and various components that manage the various features of the simulation.
   The componets shown in medium-blue are user-configurable and -creatable.

The GridWorld Simulation Framework utilizes a modular design that allows users
to create new features and plug them in as components of the simulation. Every component
inherits from the :ref:`GridWorldBaseComponent <api_gridworld_base>` class and has a reference to
a :ref:`Grid <gridworld_grid>` and a dictionary of :ref:`Agents <gridworld_agent>`.
These components make up a :ref:`GridWorldSimulation <api_gridworld_sim>`, which extends the
:ref:`AgentBasedSimulation <abs>` interface. For example, a simulation might look
something like this:

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
       ...

.. _gridworld_agent:

Agent
`````

Every entity in the simulation is a :ref:`GridWorldAgent <api_gridworld_agent>`
(e.g. walls, foragers, resources, fighters, etc.). GridWorldAgents are :ref:`PrincipleAgents <api_agent>` with specific parameters
that work with their respective components. Agents must be given
an `encoding`, which is a positive integer that correlates to the type of agent and simplifies
the logic for many components of the framework. GridWorldAgents can also be configured
with an :ref:`initial position <gridworld_position>`, the ability to
:ref:`block <gridworld_blocking>` other agents' abilities, and visualization
parameters such as `shape` and `color`.

Following the dataclass model, additional agent classes can be defined that allow
them to work with various components. For example, :ref:`GridObservingAgents <gridworld_single_observer>` can work with
:ref:`Observers <gridworld_single_observer>`, and :ref:`MovingAgents <gridworld_movement>` can work with the :ref:`MoveActor <gridworld_movement>`. Any new agent class should
inhert from :ref:`GridWorldAgent <api_gridworld_agent>` and possibly from :ref:`ActingAgent <api_acting_agent>` or :ref:`ObservingAgent <api_observing_agent>` as needed.
For example, one can define a new type of agent like so:

.. code-block:: python

   from abmarl.sim.gridworld.agent import GridWorldAgent
   from abmarl.sim import ActingAgent

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
The :ref:`Grid <api_gridworld_grid>` stores :ref:`Agents <gridworld_agent>` in a two-dimensional numpy array. The Grid is configured
to be a certain size (rows and columns) and to allow types of Agents to overlap
(occupy the same cell). For example, you may want a ForagingAgent to be able to overlap
with a ResourceAgent but not a WallAgent. The `overlapping` parameter
is a dictionary that maps the Agent's `encoding` to a list of other Agents' `encodings`
with which it can overlap. For example,

.. code-block:: python

   from abmarl.sim.gridworld.grid import Grid

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

Interaction between simulation components and the :ref:`Grid <api_gridworld_grid>` is
`data open`, which means that we allow components to access the internals of the
Grid. Although this is possible and sometimes necessary, the Grid also provides
an interface for safer interactions with components. Components can `query` the
Grid to see if an agent can be placed at a specific position. Components can `place`
agents at a specific position in the Grid, which will succeed if that cell is available
to the agent as per the `overlapping` configuration. And Components can `remove`
agents from specific positions in the Grid. 


.. _gridworld_state:

State
`````

:ref:`State Components <api_gridworld_statebase>` manage the state of the simulation alongside the :ref:`Grid <gridworld_grid>`.
At the bare minimum, each State resets the part of the simulation that it manages
at the the start of each episode.


.. _gridworld_actor:

Actor
`````

:ref:`Actor Components <api_gridworld_actor>` are responsible for processing agent actions and producing changes
to the state of the simulation. Actors assign supported agents with an appropriate
action space and process agents' actions based on the Actor's key. The result of
the action is a change in the simulation's state, and Actors should return that
change in a reasonable form. For example, the :ref:`MoveActor <gridworld_movement>` appends :ref:`MovingAgents' <gridworld_movement>` action
spaces with a 'move' channel and looks for the 'move' key in the agent's incoming
action. After a move is processed, the MoveActor returns if the move was successful.


.. _gridworld_observer:

Observer
````````

:ref:`Observer Components <api_gridworld_observer>` are responsible for creating an
agent's observation of the state of the simulation. Observers assign supported agents
with an appropriate observation space and generate observations based on the
Observer's key. For example, the :ref:`SingleGridObserver <gridworld_single_observer>` generates an observation of the nearby grid and
stores it in the 'grid' channel of the :ref:`ObservingAgent's <gridworld_single_observer>` observation.


.. _gridworld_done:

Done
````

:ref:`Done Components <api_gridworld_done>` manage the "done state" of each agent and of the simulation as a
whole. Agents that are reported as done will cease sending actions to the simulation, 
and the episode will end when all the agents are done or when the simulation is done.


.. _gridworld_wrappers:

Component Wrappers
``````````````````

The GridWorld Simulation Framework also supports
:ref:`Component Wrappers <api_gridworld_wrappers>`. Wrapping a component
can be useful when you don't want to add a completely new component and
only need to make a modification to the way a component already works. A component
wrapper is itself a component, and so it must implement the same interface as the
wrapped component to ensure that it works within the framework. A component wrapper
also defines additional functions for wrapping spaces and data to and from those
spaces: ``check_space`` for ensuring the space can be transformed, ``wrap_space`` to
perform the transformation, and ``wrap_point`` to map data to the transformed space.

As its name suggests, a :ref:`Component Wrapper <api_gridworld_wrappers>` stands
between the underlying component and other
objects with which it exchanges data. As such, a wrapper typically modifies
the incoming/outgoing data before leveraging the underlying component for
the actual datda processing. The main difference among wrapper types is in
the direction of data flow, which we detail below.

Actor Wrappers
~~~~~~~~~~~~~~

An :ref:`Actor Wrappers <api_gridworld_actor_wrappers>` receives actions in the
`wrapped_space` through the ``process_action``
function. It can modify the data before sending it to the underlying Actor to
process. An Actor Wrapper may need to modify the action spaces of corresponding agents
to ensure that the action arrives in the correct format. 


.. _gridworld_built_in_features:

Built-in Features
-----------------

Below is a list of some features that are available to use out of the box. Rememeber,
you can create your own features in
the GridWorld Simulation Framework and use many combinations of components together
to make up a simulation.


.. _gridworld_position:

Position
````````

:ref:`Agents <gridworld_agent>` have `positions` in the :ref:`Grid <gridworld_grid>` that are managed by the
:ref:`PositionState <api_gridworld_state_position>`. Agents
can be configured with an `initial position`, which is where they will start at the
beginning of each episode. If they are not given an `initial position`, then they
will start at a random cell in the grid. Agents can overlap according to the
:ref:`Grid's <gridworld_grid>` `overlapping` configuration. For example, consider the following setup:

.. code-block:: python

   import numpy as np
   from abmarl.sim.gridworld.agent import GridWorldAgent
   from abmarl.sim.gridworld.grid import Grid
   from abmarl.sim.gridworld.state import PositionState

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

`agent0` is configured with an `initial position` and `agent1` is not. At the
start of each episode, `agent0` will be placed at (2, 4) and `agent1` will be placed
anywhere in the grid (except for (2,4) because they cannot overlap).

.. figure:: .images/gridworld_positioning.png
   :width: 100 %
   :alt: Agents starting positions

   agent0 in green starts at the same cell in every episode, and agent1 in blue
   starts at a random cell each time.


.. _gridworld_movement:

Movement
````````

:ref:`MovingAgents <api_gridworld_agent_moving>` can move around the
:ref:`Grid <gridworld_grid>` in conjunction with the
:ref:`MoveActor <api_gridworld_actor_move>`. MovingAgents require a `move range`
parameter, indicating how many spaces away they can move in a single step. Agents
cannot move out of bounds and can only move to the same cell as another agent if
they are allowed to overlap. For example, in this setup

.. code-block:: python

   import numpy as np
   from abmarl.sim.gridworld.agent import MovingAgent
   from abmarl.sim.gridworld.grid import Grid
   from abmarl.sim.gridworld.state import PositionState
   from abmarl.sim.gridworld.actor import MoveActor

   agents = {
       'agent0': MovingAgent(
           id='agent0', encoding=1, move_range=1, initial_position=np.array([2, 2])
       ),
       'agent1': MovingAgent(
           id='agent1', encoding=1, move_range=2, initial_position=np.array([0, 2])
       )
   }
   grid = Grid(5, 5, overlapping={1: [1]})
   position_state = PositionState(agents=agents, grid=grid)
   move_actor = MoveActor(agents=agents, grid=grid)

   position_state.reset()
   move_actor.process_action(agents['agent0'], {'move': np.array([0, 1])})
   move_actor.process_action(agents['agent1'], {'move': np.array([2, 1])})

`agent0` starts at position (2, 2) and can move up to one cell away. `agent1`
starts at (0, 2) and can move up to two cells away. The two agents can overlap
each other, so when the move actor processes their actions, both agents will be
at position (2, 3).

.. figure:: .images/gridworld_movement.png
   :width: 100 %
   :alt: Agents moving in the grid

   agent0 and agent1 move to the same cell.

The :ref:`MoveActor <api_gridworld_actor_move>` automatically assigns a `null action`
of `[0, 0]`, indicating no move.


.. _gridworld_single_observer:

Single Grid Observer
````````````````````

:ref:`GridObservingAgents <api_gridworld_agent_observing>` can observe the state of the :ref:`Grid <gridworld_grid>` around them, namely which
other agents are nearby, via the :ref:`SingleGridObserver <api_gridworld_observer_single>`. The SingleGridObserver generates
a two-dimensional array sized by the agent's `view range` with the observing
agent located at the center of the array. All other agents within the `view range` will
appear in the observation, shown as their `encoding`. For example, the following setup

.. code-block:: python

   import numpy as np
   from abmarl.sim.gridworld.agent import GridObservingAgent, GridWorldAgent
   from abmarl.sim.gridworld.grid import Grid
   from abmarl.sim.gridworld.state import PositionState
   from abmarl.sim.gridworld.observer import SingleGridObserver

   agents = {
       'agent0': GridObservingAgent(id='agent0', encoding=1, initial_position=np.array([2, 2]), view_range=3),
       'agent1': GridWorldAgent(id='agent1', encoding=2, initial_position=np.array([0, 1])),
       'agent2': GridWorldAgent(id='agent2', encoding=3, initial_position=np.array([1, 0])),
       'agent3': GridWorldAgent(id='agent3', encoding=4, initial_position=np.array([4, 4])),
       'agent4': GridWorldAgent(id='agent4', encoding=5, initial_position=np.array([4, 4])),
       'agent5': GridWorldAgent(id='agent5', encoding=6, initial_position=np.array([5, 5]))
   }
   grid = Grid(6, 6, overlapping={4: [5], 5: [4]})
   position_state = PositionState(agents=agents, grid=grid)
   observer = SingleGridObserver(agents=agents, grid=grid)

   position_state.reset()
   observer.get_obs(agents['agent0'])

will position agents as below and output an observation for `agent0` (blue) like so:

.. figure:: .images/gridworld_observation.png
   :width: 50 %

.. code-block::

   [-1, -1, -1, -1, -1, -1, -1],
   [-1,  0,  2,  0,  0,  0,  0],
   [-1,  3,  0,  0,  0,  0,  0],
   [-1,  0,  0,  1,  0,  0,  0],
   [-1,  0,  0,  0,  0,  0,  0],
   [-1,  0,  0,  0,  0, 4*,  0],
   [-1,  0,  0,  0,  0,  0,  6]

Since `view range` is the number of cells away that can be observed, the observation size is
``(2 * view_range + 1) x (2 * view_range + 1)``. `agent0` is centered in the middle
of this array, shown by its `encoding`: 1. All other agents appear in the observation
relative to `agent0's` position and shown by their `encodings`. The agent observes some out
of bounds cells, which appear as -1s. `agent3` and `agent4` occupy the same cell,
and the :ref:`SingleGridObserver <api_gridworld_observer_single>` will randomly select between their `encodings`
for the observation.

By setting `observe_self` to False, the :ref:`SingleGridObserver <api_gridworld_observer_single>`
can be configured so that an agent doesn't observe itself and only observes
other agents, which may be helpful if overlapping is an important part of the simulation.

The :ref:`SingleGridObserver <api_gridworld_observer_single>` automatically assigns
a `null observation` as a view matrix of all -2s, indicating that everything is
masked.

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
cell. Any cell whose center falls between those two lines will be masked, as shown below.

.. figure:: .images/gridworld_blocking.png
   :width: 100 %
   :alt: Masked cells from blocking agent

   The black agent is a wall agent that masks part of the grid from the blue agent.
   Cells whose centers fall betweent the lines are masked. Centers that fall directly
   on the line or outside of the lines are not masked. Two setups are shown to 
   demonstrate how the masking may change based on the agents' positions.


Multi Grid Observer
```````````````````

Similar to the :ref:`SingleGridObserver <api_gridworld_observer_single>`, the :ref:`MultiGridObserver <api_gridworld_observer_multi>` displays a separate array
for every `encoding`. Each array shows the relative positions of the agents and the
number of those agents that occupy each cell. Out of bounds indicators (-1) and
masked cells (-2) are present in every grid. For example, this setup would
show an observation like so:

.. figure:: .images/gridworld_observation.png
   :width: 50 %

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

:ref:`MultiGridObserver <api_gridworld_observer_multi>` may be preferable to
:ref:`SingleGridObserver <api_gridworld_observer_single>` in simulations where
there are many overlapping agents.

The :ref:`MultiGridObserver <api_gridworld_observer_multi>` automatically assigns
a `null observation` of a tensor of all -2s, indicating that everything is masked.

Health
``````

:ref:`HealthAgents <api_gridworld_agent_health>` track their `health` throughout the simulation. `Health` is always bounded
between 0 and 1. Agents whose `health` falls to 0 are marked as `inactive`. They can be given an
`initial health`, which they start with at the beginning of the episode. Otherwise,
their `health` will be a random number between 0 and 1, as managed by the :ref:`HealthState <api_gridworld_state_health>`.
Consider the following setup:

.. code-block:: python

   from abmarl.sim.gridworld.agent import HealthAgent
   from abmarl.sim.gridworld.grid import Grid
   from abmarl.sim.gridworld.state import HealthState

   agent0 = HealthAgent(id='agent0', encoding=1)
   grid = Grid(3, 3)
   agents = {'agent0': agent0}
   health_state = HealthState(agents=agents, grid=grid)
   health_state.reset()

`agent0` will be assigned a random `health` value between 0 and 1.


Attacking
`````````

`Health` becomes more interesting when we let agents attack one another.
:ref:`AttackingAgents <api_gridworld_agent_attack>` work in conjunction with 
an :ref:`AttackActor <api_gridworld_actor_attack>`. They have an `attack range`, which dictates
the range of their attack; an `attack accuracy`, which dictates the chances of the
attack being successful; an `attack strength`, which dictates how much `health`
is depleted from the attacked agent, and an `attack count`, which dictates the
number of attacks an agent can make per turn.

An AttackActor interprets these properties and processes attack dynamics according
to its own internal design. In general, each AttackActor determines some set of
attackable agents according to the following criteria:

   #. The `attack mapping`, which is a dictionary that determines which `encodings`
      can attack other `encodings` (similar to the `overlapping` parameter for the
      :ref:`Grid <gridworld_grid>`), must allow the attack.
   #. The relative positions of the two agents must fall within the attacking agent's
      `attack range`.
   #. The attackable agent must not be masked (e.g. hiding behind a wall). The masking
      is determined the same way as view blocking described above.
   #. Any additional criteria for the specific Attack Actor.

Then, the AttackActor selects agents from that set based on the attacking agent's `attack count`.
When an agent is successfully attacked, its health is depleted by the
attacking agent's `attack strength`, which may result in the attacked agent's death. AttackActors
can be configured to allow multiple attacks against a single agent per attacking
agent and per turn via the `stacked attacks` property. The following four AttackActors
are built into Abmarl:

Binary Attack Actor
~~~~~~~~~~~~~~~~~~~

Under the :ref:`BinaryAttackActor <api_gridworld_actor_binary_attack>`,
:ref:`AttackingAgents <api_gridworld_agent_attack>` can choose to use some number
of their attacks or not to attack at all. For each attack, the Binary Attack Actor
randomly searches the vicinity of the attacking agent for an attackble agent according to
the basic criteria listed above. Consider the following setup:

.. code-block:: python

   import numpy as np
   from abmarl.sim.gridworld.agent import AttackingAgent, HealthAgent
   from abmarl.sim.gridworld.grid import Grid
   from abmarl.sim.gridworld.state import PositionState, HealthState
   from abmarl.sim.gridworld.actor import BinaryAttackActor

   agents = {
       'agent0': AttackingAgent(
           id='agent0',
           encoding=1,
           initial_position=np.array([0, 0]),
           attack_range=1,
           attack_strength=0.4,
           attack_accuracy=1,
           attack_count=2
       ),
       'agent1': HealthAgent(id='agent1', encoding=2, initial_position=np.array([1, 0]), initial_health=1),
       'agent2': HealthAgent(id='agent2', encoding=2, initial_position=np.array([1, 1]), initial_health=0.3),
       'agent3': HealthAgent(id='agent3', encoding=3, initial_position=np.array([0, 1]))
   }
   grid = Grid(2, 2)
   position_state = PositionState(agents=agents, grid=grid)
   health_state = HealthState(agents=agents, grid=grid)
   attack_actor = BinaryAttackActor(agents=agents, grid=grid, attack_mapping={1: [2]}, stacked_attacks=False)

   position_state.reset()
   health_state.reset()
   attack_actor.process_action(agents['agent0'], {'attack': 2})
   assert not agents['agent2'].active
   assert agents['agent1'].active
   assert agents['agent3'].active
   attack_actor.process_action(agents['agent0'], {'attack': 2})
   assert agents['agent1'].active
   assert agents['agent3'].active


As per the `attack mapping`, `agent0` can attack `agent1` or `agent2` but not
`agent3`. It can make two attacks per turn, but because the `stacked attacks` property
is False, it cannot attack the same agent twice in the same turn. Looking at the
`attack strength` and `initial health` of the agents, we can see that `agent0`
should be able to kill `agent2` with one attack but it will require three attacks
to kill `agent1`. In each turn, `agent0` uses both of its attacks. In the first
turn, both `agent1` and `agent2` are attacked and `agent2` dies. In the second
turn, `agent0` attempts two attacks again, but because there is only one attackable
agent in its vicinity and because `stacked attacks` are not allowed, only one of
its attacks is successful: `agent1` is attacked, but it continues to live since
it still has health. `agent3` was never attacked because although it is within
`agent0`'s `attack range`, it is not in the `attack mapping`.

The :ref:`BinaryAttackActor <api_gridworld_actor_binary_attack>` automatically
assigns a `null action` of 0, indicating no attack.


Encoding Based Attack Actor
~~~~~~~~~~~~~~~~~~~~~~~~~~~



SelectiveAttackActor
~~~~~~~~~~~~~~~~~~~~

RestrictedSelectiveAttackActor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






RavelActionWrapper
``````````````````

The :ref:`RavelActionWrapper <api_gridworld_ravel_action_wrappers>` transforms
Discrete, MultiBinary, MultiDiscrete, bounded integer Box, and any nesting of those
spaces into a Discrete space by "ravelling" their values according to numpy's
``ravel_multi_index`` function. Thus, actions that are represented by arrays are
converted into unique Discrete numbers. For example, we can apply the RavelActionWrapper
to the MoveActor, like so:

.. code-block:: python

   from abmarl.sim.gridworld.agent import MovingAgent
   from abmarl.sim.gridworld.grid import Grid
   from abmarl.sim.gridworld.state import PositionState
   from abmarl.sim.gridworld.actor import MoveActor
   from abmarl.sim.gridworld.wrapper import RavelActionWrapper
   
   agents = {
       'agent0': MovingAgent(id='agent0', encoding=1, move_range=1),
       'agent1': MovingAgent(id='agent1', encoding=1, move_range=2)
   }
   grid = Grid(5, 5)
   position_state = PositionState(agents=agents, grid=grid)
   move_actor = MoveActor(agents=agents, grid=grid)
   for agent in agents.values():
       agent.finalize()
   position_state.reset()

   # Move actor without wrapper
   actions = {
       agent.id: agent.action_space.sample() for agent in agents.values()
   }
   print(actions)
   # >>> {'agent0': OrderedDict([('move', array([1, 1]))]), 'agent1': OrderedDict([('move', array([ 2, -1]))])}
   
   # Wrapped move actor
   move_actor = RavelActionWrapper(move_actor)
   actions = {
       agent.id: agent.action_space.sample() for agent in agents.values()
   }
   print(actions)
   # >>> {'agent0': OrderedDict([('move', 1)]), 'agent1': OrderedDict([('move', 22)])}

The actions from the unwrapped actor are in the original `Box` space, whereas after
we apply the wrapper, the actions from the wrapped actor are in the transformed
`Discrete` space. The actor will receive move actions in the `Discrete` space and convert
them to the `Box` space before passing them to the MoveActor.
