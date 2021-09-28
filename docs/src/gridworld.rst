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
inherits from the GridWorldBaseComponent class and has a reference to the grid and the dictionary
of agents.

A GridWorldSimulation is composed of a dictionary of Agents, a Grid, and various
Components. It follows the AgentBasedSimulation interface and relies on the components
themselves to implement the pieces of the interface. For example, a simulation might
look something like

.. code-block:: python

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

FIGURE ### shows a visual depiction of the framework being used to create a simulation.
See THIS TUTORIAL for an indepth example.

GridWorldSimulation also provides two builders: (1) build sim and (2) build sim
from file. See THIS TUTORIAL for information on how to use these builders.


Agent
`````

Every entity in the GridWorld is a GridWorldAgent (e.g. walls, foragers, resources, fighters, etc.).
GridWorldAgents are PrincipalAgents with specific parameters needed to be used in
a GridWorld Simulation. In particular, agents must be given an encoding, which is
an integer that defines the type of agent and simplifies the logic for many components
of the framework. GridWorldAgents can also be configured with an initial position,
the ability to block other agents' abilities, and rendering parameters such as shape
and color.

Following the dataclass model, additional agent classes can be defined that allow
agents to work with various components. For example, ObservingAgents can work with
Observers and MovingAgents can work with the MoveActor. Any new agent class should
inhert from GridWorldAgent and possibly from ActingAgent or ObservingAgent as needed.
For example, one can define a new type of agent like so:

.. code-block:: python

   class CommunicatingAgent(GridWorldAgent, ActingAgent):
       def __init__(self, broadcast_range=None, **kwargs):
           super().__init__(**kwargs)
           self.broadcast_range = broadcast_range

.. WARNING::
   Agents should follow the dataclass model, meaning that they should only be given
   parameters. All functionality should be written in the Components that work with
   the agents.

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


State
`````

State Components manage the state of the simulation alongside the Grid. Each State
has a reset function that resets the simulation at the the start of each episode.

Actor
`````

Actor Components are responsible for processing agent actions and producing changes
to the state of the simulation. Actors assign supported agents with an appropriate
action space and process agents' actions based on the Actor's key. For example, the
MoveActor appends MovingAgents' action spaces with a 'move' channel and look for
the 'move' key in the agent's incoming action.

Observer
````````

Similar to Actor Components, Observer Components are responsible for creating an
agent's observation of the state of the simulation. Observers assign supported agents
with an appropriate observation space and generate observations based on the
Observer's key. For example, the SingleGridObserver generates an observation and
stores it in the 'grid' channel of the agent's observation.

Done
````

Done Components manage the "done state" of each agent and of the simulation as a
whole via their ``get_done`` and ``get_all_done`` interface. Agents that are reported
as done will cease sending actions to the simulation.


Features
--------



