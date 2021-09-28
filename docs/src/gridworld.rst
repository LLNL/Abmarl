.. Abmarl gridworld documentation

GridWorld Simulation Framework
==============================

Abmarl provides a GridWorld Simulation Framework for setting up varieties of 
Agent Based Simulations, which can be connected to Reinforcement Learning algorithms
through Abmarl's AgentBasedSimulation interface.

Framework Design
----------------

The GridWorld Simulation Framework utilizes a modular design that allows developers
to create new features and plug them in as components of the simulation. Every component
inherits from the Component Base class and has a reference to the grid and the dictionary
of agents.

The components are then fit together in the simulation's initialization...

Agent
`````

Grid
````
The Grid stores agents in a two-dimensional numpy array. The Grid is configured
to be a certain size (rows and columns) and to allow types of agents to overlap
(occupy the same cell). For example, you may want a Foraging Agent to be able to overlap
with a ResourceAgent but not a WallAgent. The overlapping argument
is a dictionary that maps the agent's encoding to a list of other agents' encodings
with which it can overlap. For example,

.. code-block:: python

   overlapping = {
       1: [2],
       2: [1, 3],
       3: [2, 3]
   }
   grid = Grid(5, 6, overlapping=overlapping)

means that agents whose encoding is 1 can overlap with other agents whose encoding
is 2; agents whose encoding is 2 can overlap with other agents whose encodings are
1 or 3; and agents whose encoding is 3 can overlap with other agents whose encodings
are 2 or 3.

.. WARNING::
   To avoid undefined behavior, the overlapping should be symmetric, so that if
   2 can overlap with 3, then 3 can also overlap with 2.

.. NOTE::
   If overlapping is not provided, then no agents will be able to occupy the same
   cell in the Grid.

Interaction between simulation components (see below) and the grid is
`data open`, which means that we allow components to access the internals of the
grid. Although this is possible and sometimes necessary, the Grid also provides
an interface for safer interactions with components.

Components can ``query`` the Grid to see if an agent can be placed at a specific location.
Components can ``place`` agents at a specific location in the Grid, which will succeed
if that cell is available to the agent as per the overlapping configuration. And
Components can ``remove`` agents from specific locations in the Grid. 

Base Component
``````````````

State
`````

Actor
`````

Observer
````````

Done
````


Features
--------



