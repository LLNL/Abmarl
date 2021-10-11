
.. Abmarl documentation GridWorld tutorial.

GridWorld
=========

GridWorld Simulation Framework is composed of feature modules that fit together
to allow users to create a variety of simulations using the same pieces and to easily
design their own features. In this tutorial, we demostrate how
to use the GridWorld Simulation Framework to create a multi-team battle simulation.
We then show how the exact same modules can be used reconfigured to create a maze-navigation
simulation. Finally, we show how easy it is to add custom features as modules and
plug them into the simulation framework.

Team Battle
-----------

The team battle scenario involves multiple teams fighting against each other.
The goal of each team is to be the last team alive, at which point the simulation will end.
Each agent can move around the grid and attack agents from other teams. Each agent
can observe the grid around its position. We will reward each agent for successful
kills and penalize them for bad moves. This simulation can be found in full
`in our repo <>`_.

First, we import the pieces that we will need. Each feature is already in Abmarl,
so we don't need to create anything new.

.. code-block:: python

   from abmarl.sim.gridworld.base import GridWorldSimulation
   from abmarl.sim.gridworld.agent import GridObservingAgent, MovingAgent, AttackingAgent, HealthAgent
   from abmarl.sim.gridworld.state import HealthState, PositionState
   from abmarl.sim.gridworld.actor import MoveActor, AttackActor
   from abmarl.sim.gridworld.observer import SingleGridObserver
   from abmarl.sim.gridworld.done import OneTeamRemainingDone
   from abmarl.tools.matplotlib_utils import mscatter # Needed for nice renderings

Then, we define our agent types. This simulation will only have a single type:
the BattleAgent. Most of the agent attributes will be the same, and we can preconfigure
that in the class definition so we don't have to do it for every agent.

.. code-block:: python

   class BattleAgent(GridObservingAgent, MovingAgent, AttackingAgent, HealthAgent):
       def __init__(self, **kwargs):
           super().__init__(
               move_range=1,
               attack_range=1,
               attack_strength=1,
               attack_accuracy=1,
               view_range=3,
               **kwargs
           )

Having defined the BattleAgent, we then put all the components together into a single
simulation: TeamBattleSim.

.. code-block:: python

   class TeamBattleSim(GridWorldSimulation):
       def __init__(self, **kwargs):
           self.agents = kwargs['agents']
   
           # State Components
           self.position_state = PositionState(**kwargs)
           self.health_state = HealthState(**kwargs)
   
           # Action Components
           self.move_actor = MoveActor(**kwargs)
           self.attack_actor = AttackActor(**kwargs)
   
           # Observation Components
           self.grid_observer = SingleGridObserver(**kwargs)
   
           # Done Compoennts
           self.done = OneTeamRemainingDone(**kwargs)
           
           self.finalize()

Next we define the start state of each simulation. We lean on the State Components
to perform the reset. Note that we must track the rewards explicitly.

.. code-block:: python

   class TeamBattleSim(GridWorldSimulation):
       ...
 
       def reset(self, **kwargs):
           self.position_state.reset(**kwargs)
           self.health_state.reset(**kwargs)
        
           # Track the rewards
           self.rewards = {agent.id: 0 for agent in self.agents.values()}

Then we define how the simulation will step forward, leaning on the Actors to process
their part of the action. The Actors' results are used to determine the agents'
rewards.

.. code-block:: python

   class TeamBattleSim(GridWorldSimulation):
       ...

       def step(self, action_dict, **kwargs):
           # Process attacks:
           for agent_id, action in action_dict.items():
               agent = self.agents[agent_id]
               attacked_agent = self.attack_actor.process_action(agent, action, **kwargs)
               if attacked_agent is not None:
                   self.rewards[attacked_agent.id] -= 1
                   self.rewards[agent.id] += 1
               else:
                   self.rewards[agent.id] -= 0.1
   
           # Process moves
           for agent_id, action in action_dict.items():
               agent = self.agents[agent_id]
               if agent.active:
                   move_result = self.move_actor.process_action(agent, action, **kwargs)
                   if not move_result:
                       self.rewards[agent.id] -= 0.1
           
           # Entropy penalty
           for agent_id in action_dict:
               self.rewards[agent_id] -= 0.01

Then we define each of the getters.

.. code-block:: python

   class TeamBattleSim(GridWorldSimulation):
       ...

       def get_obs(self, agent_id, **kwargs):
           agent = self.agents[agent_id]
           return {
               **self.grid_observer.get_obs(agent, **kwargs)
           }
   
       def get_reward(self, agent_id, **kwargs):
           reward = self.rewards[agent_id]
           self.rewards[agent_id] = 0
           return reward
   
       def get_done(self, agent_id, **kwargs):
           return self.done.get_done(self.agents[agent_id])
   
       def get_all_done(self, **kwargs):
           return self.done.get_all_done(**kwargs)
   
       def get_info(self, agent_id, **kwargs):
           return {}

Finally, in order to visualize our simulation, we define a render function.

.. code-block:: python

   class TeamBattleSim(GridWorldSimulation):
       ...

       def render(self, fig=None, **kwargs):
           fig.clear()
           ax = fig.gca()
   
           # Draw the gridlines
           ax.set(xlim=(0, self.position_state.cols), ylim=(0, self.position_state.rows))
           ax.set_xticks(np.arange(0, self.position_state.cols, 1))
           ax.set_yticks(np.arange(0, self.position_state.rows, 1))
           ax.grid()
   
           # Draw the agents
           agents_x = [
               agent.position[1] + 0.5 for agent in self.agents.values() if agent.active
           ]
           agents_y = [
               self.position_state.rows - 0.5 - agent.position[0]
               for agent in self.agents.values() if agent.active
           ]
           shape = [agent.render_shape for agent in self.agents.values() if agent.active]
           color = [agent.render_color for agent in self.agents.values() if agent.active]
           mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, facecolor=color)
   
           plt.plot()
           plt.pause(1e-6)

Now that we've defined our agents and simulations, let's create them and run the
simulation. First, we'll create the agents. There will be 4 teams, so we want to
color the agent by team and start them at different corners of the grid. Besides that,
all agent attributes will be the same, and here we benefit from pre-configuring
the attributes in the class definition.

.. code-block:: python

   colors = ['red', 'blue', 'green', 'gray'] # Team colors
   positions = [np.array([1,1]), np.array([1,6]), np.array([6,1]), np.array([6,6])] # Grid corners
   agents = {
       f'agent{i}': BattleAgent(
           id=f'agent{i}',
           encoding=i%4+1,
           render_color=colors[i%4],
           initial_position=positions[i%4]
       ) for i in range(24)
   }

Having created the agents, we can now build the simulation. We will allow agents
from the same team to occupy the same cell and allow agents to attack other agents
if they are on different teams.

.. code-block:: python

   overlap_map = {
       1: [1],
       2: [2],
       3: [3],
       4: [4]
   }
   attack_map = {
       1: [2, 3, 4],
       2: [1, 3, 4],
       3: [1, 2, 4],
       4: [1, 2, 3]
   }
   sim = TeamBattleSim.build_sim(
       8, 8,
       agents=agents,
       overlapping=overlap_map,
       attack_mapping=attack_map
   )

Finally, we can run the simulation with random actions and visualize it.

.. code-block:: python

   sim.reset()
   fig = plt.figure()
   sim.render(fig=fig)
   
   from pprint import pprint
   for i in range(50):
       action = {
           agent.id: agent.action_space.sample() for agent in agents.values()
       }
       sim.step(action)
       sim.render(fig=fig)

Maze Navigation
---------------

Using the same GridWorld modules as above, we can create a Maze Navigation simulation.
The Maze Navigation Simulation will contain a single moving agent navigating a maze
defined by wall agents in the grid. The moving agents goal is to reach a target
agent. We will construct the Grid by reading a grid file.

.. NOTE::

   This simulation is really a single-agent simulation. While we have multiple entities
   like walls and a target agent, the only agent that is actually doing something
   is the navigation agent. We will use some custom modifications to make the single
   agent case easier, showing that we can use our components in a single agent
   simulation with custom modifications.

First we import the components that we need. Each feature is already in Abmarl, and
they are the same features that we used in the previous tutorial.

.. code-block:: python

   from matplotlib import pyplot as plt
   import numpy as np
   
   from abmarl.sim.gridworld.base import GridWorldSimulation
   from abmarl.sim.gridworld.agent import GridObservingAgent, MovingAgent, GridWorldAgent
   from abmarl.sim.gridworld.state import PositionState
   from abmarl.sim.gridworld.actor import MoveActor
   from abmarl.sim.gridworld.observer import SingleGridObserver
   from abmarl.tools.matplotlib_utils import mscatter

Then, we define our agent types. We need an MazeNavigationAgent, WallAgents to act
as the barriers of the maze, and a TargetAgent to indicate the goal. Although we
have these three types, we only need to define the MazeNavigationAgent because
the WallAgent and the TargetAgent are the same as a generic GridWorldAgent.

.. code-block:: python

   class MazeNavigationAgent(GridObservingAgent, MovingAgent):
       def __init__(self, **kwargs):
           super().__init__(move_range=1, **kwargs)

Here we have pre-configured the agent with a move_range of 1 becuase that makes
the most sense for navigating mazes, but we have not pre-configured the ``view_range``
since that is a parameter we may want to adjust, and it is easier to adjust it
at the agent's initialization.

Then, we define the simulation using the GridWorld components and define all the
necessary functions. We find it convient to explicitly store a reference to the
navigation agent and the target agent. We've also taken several shortcuts because
we are a single-agent simulation. Finally, rather than defining a new component
for our very simply done condition, we just write the condition itself in the function.

.. code-block:: python

   class MazeNaviationSim(GridWorldSimulation):
       def __init__(self, **kwargs):
           self.agents = kwargs['agents']

           # Store the navigation and target agents
           self.navigator = kwargs['agents']['navigator']
           self.target = kwargs['agents']['target']
   
           # State Components
           self.position_state = PositionState(**kwargs)
   
           # Action Components
           self.move_actor = MoveActor(**kwargs)
   
           # Observation Components
           self.grid_observer = SingleGridObserver(**kwargs)
   
           self.finalize()
   
       def reset(self, **kwargs):
           self.position_state.reset(**kwargs)
   
           # Since there is only one agent that produces actions, there is only one reward.
           self.reward = 0
       
       def step(self, action_dict, **kwargs):    
           # Only the navigation agent will send actions, so we pull that out
           action = action_dict['navigator']
           move_result = self.move_actor.process_action(self.navigator, action, **kwargs)
           if not move_result:
               self.reward -= 0.1
           
           # Entropy penalty
           self.reward -= 0.01
       
       def render(self, fig=None, **kwargs):
           fig.clear()
           ax = fig.gca()
   
           # Draw the gridlines
           ax.set(xlim=(0, self.position_state.cols), ylim=(0, self.position_state.rows))
           ax.set_xticks(np.arange(0, self.position_state.cols, 1))
           ax.set_yticks(np.arange(0, self.position_state.rows, 1))
           ax.grid()
   
           # Draw the agents
           agents_x = [
               agent.position[1] + 0.5 for agent in self.agents.values() if agent.active
           ]
           agents_y = [
               self.position_state.rows - 0.5 - agent.position[0]
               for agent in self.agents.values() if agent.active
           ]
           shape = [agent.render_shape for agent in self.agents.values() if agent.active]
           color = [agent.render_color for agent in self.agents.values() if agent.active]
           mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, facecolor=color)
   
           plt.plot()
           plt.pause(1e-6)
   
       def get_obs(self, agent_id, **kwargs):
           # pass the navigation agent itself to the observer becuase it is the only
           # agent that takes observations
           return {
               **self.grid_observer.get_obs(self.navigator, **kwargs)
           }
   
       def get_reward(self, agent_id, **kwargs):
           # Custom reward function
           if self.get_all_done():
               self.reward = 1
           reward = self.reward
           self.reward = 0
           return reward
   
       def get_done(self, agent_id, **kwargs):
           return self.get_all_done()
   
       def get_all_done(self, **kwargs):
           # We define the done condition here directly rather than creating a
           # separate component for it.
           return np.all(self.navigator.position == self.target.position)
   
       def get_info(self, agent_id, **kwargs):
           return {}

With everything defined, we're ready to create and run our simulation. We will
create the simulation by reading a simulation file that shows the positions of
each agent type in the grid. We will use <maze.txt>, which looks like this:

.. code-block::

   0 0 0 0 W 0 W W 0 W W 0 0 W W 0 W 0
   W 0 W 0 N 0 0 0 0 0 W 0 W W 0 0 0 0
   W W W W 0 W W 0 W 0 0 0 0 W W 0 W W
   0 W 0 0 0 W W 0 W 0 W W 0 0 0 0 0 0
   0 0 0 W 0 0 W W W 0 W 0 0 W 0 W W 0
   W W W W 0 W W W W W W W 0 W 0 T W 0
   0 0 0 0 0 W 0 0 0 0 0 0 0 W 0 W W 0
   0 W 0 W 0 W W W 0 W W 0 W W 0 W 0 0

In order to assign meaning to the values in the grid file, we must create an object
registry that maps the values in the files to objects. We will use ``W`` for WallAgents,
``N`` for the Navigation Agent, and ``T`` for the TargetAgent. The values of the
object registry must be lambda functions that take in a value and produce an agent.
See <> for more detail on the object_registry.

.. code-block:: python

   object_registry = {
       'N': lambda n: MazeNavigationAgent(
           id=f'navigator',
           encoding=1,
           view_range=2, # Observation parameter that we can adjust as desired
           render_color='blue',
       ),
       'T': lambda n: GridWorldAgent(
           id=f'target',
           encoding=3,
           render_color='green'
       ),
       'W': lambda n: GridWorldAgent(
           id=f'wall{n}',
           encoding=2,
           view_blocking=True,
           render_shape='s'
       )
   }

Now we can create the simulation from the maze file using the object registry.
We must allow the navigation agent and the target agent to overlap since that is
our done condition, and without it the simulation would never end.

.. code-block:: python

   file_name = 'maze.txt'
   sim = MazeNaviationSim.build_sim_from_file(
       file_name,
       object_registry,
       overlapping={1: [3], 3: [1]}
   )
   sim.reset()
   fig = plt.figure()
   sim.render(fig=fig)
  
   for i in range(100):
       action = {'navigator': sim.navigator.action_space.sample()}
       sim.step(action)
       sim.render(fig=fig)
       done = sim.get_all_done()
       if done:
           plt.pause(1)
           break

We can examine the observation to see how the view blocking walls effect what the
navigation agent can observe. In the state shown in Figure ###, the observation
is:

.. code-block::

   GRID OBSERVATION SHOWING THE EFFECTS OF VIEW BLOCKING AGENTS.

# TODO: Put an observation showing in the above tutorial with the multiple teams.

Communication Blocking
----------------------

Suppose we want to create a simulation in which some agents send messages to each
other in an attempt to reach consensus while another group of agents attempts to
block their messages. Abmarl's GridWorld Simulation Frameowkr already contains the
features for the blocking agents, but how can we add the communication features.
In this tutorial, we will show just how easy it is to create new features that can
plug into the simulation framework.

Let's start by laying the groundwork using components already in Abmarl.

.. code-block:: python
   
   from abmarl.sim.gridworld.agent import MovingAgent, GridObservingAgent
   from abmarl.sim.gridworld.base import GridWorldSimulation
   from abmarl.sim.gridworld.state import PositionState
   from abmarl.sim.gridworld.actor import MoveActor
   from abmarl.sim.gridworld.observer import SingleGridObserver

   class BlockingAgent(MovingAgent, GridObservingAgent):
       def __init__(self, **kwargs):
           super().__init__(view_blocking=True, **kwargs)

   class BroadcastSim(GridWorldSimulation):
       def __init__(self, **kwargs):
           self.agents = kwargs['agents']
           self.position_state = PositionState(**kwargs)
           self.move_actor = MoveActor(**kwargs)
           self.grid_observer = SingleGridObserver(**kwargs)
   
           self.finalize()
   
       def reset(self, **kwargs):
           self.position_state.reset(**kwargs)
           self.rewards = {agent.id: 0 for agent in self.agents.values()}
   
       def step(self, action_dict, **kwargs):   
           # process moves
           for agent_id, action in action_dict.items():
               agent = self.agents[agent_id]
               move_result = self.move_actor.process_action(agent, action, **kwargs)
               if not move_result:
                   self.rewards[agent.id] -= 0.1
   
           # Entropy penalty
           for agent_id in action_dict:
               self.rewards[agent_id] -= 0.01
       
       def render(self, fig=None, **kwargs):
           fig.clear()
           ax = fig.gca()
   
           # Draw the gridlines
           ax.set(xlim=(0, self.position_state.cols), ylim=(0, self.position_state.rows))
           ax.set_xticks(np.arange(0, self.position_state.cols, 1))
           ax.set_yticks(np.arange(0, self.position_state.rows, 1))
           ax.grid()
   
           # Draw the agents
           agents_x = [
               agent.position[1] + 0.5 for agent in self.agents.values()
           ]
           agents_y = [
               self.position_state.rows - 0.5 - agent.position[0]
               for agent in self.agents.values()
           ]
           shape = [agent.render_shape for agent in self.agents.values()]
           color = [agent.render_color for agent in self.agents.values()]
           mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, facecolor=color)
   
           plt.plot()
           plt.pause(1e-6)
       
       def get_obs(self, agent_id, **kwargs):
           agent = self.agents[agent_id]
           return {
               **self.grid_observer.get_obs(agent, **kwargs),
           }
       
       def get_reward(self, agent_id, **kwargs):
           reward = self.rewards[agent_id]
           self.rewards[agent_id] = 0
           return reward
   
       def get_done(self, agent_id, **kwargs):
           pass # Define this later
       
       def get_all_done(self, **kwargs):
           pass # Define this later
       
       def get_info(self, **kwargs):
           return {}

Now we need to build the communication pieces ourselves. We know that the GridWorld
Simulation Framework is made up of Agents, States, Actors, Observers, and Dones,
so we expect that we'll need to create each of these for our new communication feature.
Let's start with the agent.

The agent will communicate by broadcasting its message to other nearby agents.
Thus, we create a new agent with a broadcast_range and an initial_message. The
broadcast range will be used by the BroadcastActor to determine successful broadcasting,
and the initial message, an optional parameter, will be used by the BroadcastState
to set its message.

.. code-block:: python

   class BroadcastingAgent(Agent, GridWorldAgent):
       def __init__(self, broadcast_range=None, initial_message=None, **kwargs):
           super().__init__(**kwargs)
           self.broadcast_range = broadcast_range
           self.initial_message = initial_message
       
       @property
       def broadcast_range(self):
           return self._broadcast_range
       
       @broadcast_range.setter
       def broadcast_range(self, value):
           assert type(value) is int and value >= 0, "Broadcast Range must be a nonnegative integer."
           self._broadcast_range = value
       
       @property
       def initial_message(self):
           return self._initial_message
       
       @initial_message.setter
       def initial_message(self, value):
           if value is not None:
               assert -1 <= value <= 1, "Initial message must be a number between -1 and 1."
           self._initial_message = value
   
       @property
       def message(self):
           return self._message
   
       @message.setter
       def message(self, value):
           self._message = min(max(value, -1), 1)
   
       @property
       def configured(self):
           return super().configured and self.broadcast_range is not None

Next, we create the BroadcastState. This component manages the part of the simulation
state that tracks which messages have been sent among the agents. It will be used
by the BroadcastObserver to create the agent's observations. It also manages updates
to each agent's message.

.. code-block:: python

   class BroadcastingState(StateBaseComponent):
       def reset(self, **kwargs):
           for agent in self.agents.values():
               if isinstance(agent, BroadcastingAgent):
                   if agent.initial_message is not None:
                       agent.message = agent.initial_message
                   else:
                       agent.message = np.random.uniform(-1, 1)
   
           # Tracks agents receiving messages from other agents
           self.receiving_state = {
               agent.id: [] for agent in self.agents.values() if isinstance(agent, BroadcastingAgent)
           }
       
       def update_receipients(self, from_agent, to_agents):
           for agent in to_agents:
               self.receiving_state[agent.id].append((from_agent.id, from_agent.message))
   
       def update_message_and_reset_receiving(self, agent):
           receiving_from = self.receiving_state[agent.id]
           self.receiving_state[agent.id] = []
   
           messages = [message for _, message in receiving_from]
           messages.append(agent.message)
           agent.message = np.average(messages)
   
           return receiving_from

Then we define the BroadcastActor. Similar to attacking, broadcasting will be a
boolean action--either broadcast or don't broadcast. We provide a broadcast_mapping
for determine to which encodings each agent can broadcast. The message will be
successfully sent to every agent that (1) is within the broadcast range, (2) has
a compatible encoding, and (3) is not blocked from view.

.. code-block:: python

   import abmarl.sim.gridworld.utils as gu
   
   class BroadcastingActor(ActorBaseComponent):
       """
       Process sending and receiving messages between agents.
   
       Broadcasting Agents can broadcast to compatible agents within their range
       according to the broadcast mapping and if the agent is not view_blocked.
       """
       def __init__(self, broadcast_mapping=None, **kwargs):
           super().__init__(**kwargs)
           self.broadcast_mapping = broadcast_mapping
           for agent in self.agents.values():
               if isinstance(agent, self.supported_agent_type):
                   agent.action_space[self.key] = Discrete(2)
       
       @property
       def key(self):
           return 'broadcast'
       
       @property
       def supported_agent_type(self):
           return BroadcastingAgent
   
       @property
       def broadcast_mapping(self):
           """
           Dict that dictates to which agents the broadcasting agent can broadcast.
   
           The dictionary maps the broadcasting agents' encodings to a list of encodings
           to which they can broadcast. For example, the folowing broadcast_mapping:
           {
               1: [3, 4, 5],
               3: [2, 3],
           }
           means that agents whose encoding is 1 can broadcast other agents whose encodings
           are 3, 4, or 5; and agents whose encoding is 3 can broadcast other agents whose
           encodings are 2 or 3.
           """
           return self._broadcast_mapping
   
       @broadcast_mapping.setter
       def broadcast_mapping(self, value):
           assert type(value) is dict, "Broadcast mapping must be dictionary."
           for k, v in value.items():
               assert type(k) is int, "All keys in broadcast mapping must be integer."
               assert type(v) is list, "All values in broadcast mapping must be list."
               for i in v:
                   assert type(i) is int, \
                       "All elements in the broadcast mapping values must be integers."
           self._broadcast_mapping = value
   
       def process_action(self, broadcasting_agent, action_dict, **kwargs):
           """
           If the agent has chosen to broadcast, then we process their broadcast.
   
           The processing goes through a series of checks. The broadcast is successful
           if there is a receiving agent such that:
           1. The receiving agent is within range.
           2. The receiving agent is compatible according to the broadcast_mapping.
           3. The receiving agent is observable by the broadcasting agent.
           
           If the broadcast is successful, then the receiving agent receives the message
           in its observation.
           """
           def determine_broadcast(agent):
               # Generate local grid and a broadcast mask.
               local_grid, mask = gu.create_grid_and_mask(
                   agent, self.grid, agent.broadcast_range, self.agents
               )
   
               # Randomly scan the local grid for receiving agents.
               receiving_agents = []
               for r in range(2 * agent.broadcast_range + 1):
                   for c in range(2 * agent.broadcast_range + 1):
                       if mask[r, c]: # We can see this cell
                           candidate_agents = local_grid[r, c]
                           if candidate_agents is not None:
                               for other in candidate_agents.values():
                                   if other.id == agent.id: # Cannot broadcast to yourself
                                       continue
                                   elif other.encoding not in self.broadcast_mapping[agent.encoding]:
                                       # Cannot broadcast to this type of agent
                                       continue
                                   else:
                                       receiving_agents.append(other)
               return receiving_agents
   
           if isinstance(broadcasting_agent, self.supported_agent_type):
               action = action_dict[self.key]
               if action: # Agent has chosen to attack
                   return determine_broadcast(broadcasting_agent)

Then we define the BroadcastObserver. The observer enables agents to see all received
messages, including their own current message. This observer is unique from all
other components we have seen so far because it explicitly relies on the BroadcastingState
component, which we must keep in mind during initialization.

.. code-block:: python

   class BroadcastObserver(ObserverBaseComponent):
       def __init__(self, broadcasting_state=None, **kwargs):
           super().__init__(**kwargs)
   
           assert isinstance(broadcasting_state, BroadcastingState), \
               "broadcasting_state must be an instance of BroadcastingState"
           self._broadcasting_state = broadcasting_state
   
           for agent in self.agents.values():
               if isinstance(agent, self.supported_agent_type):
                   agent.observation_space[self.key] = Dict({
                       other.id: Box(-1, 1, (1,))
                       for other in self.agents.values() if isinstance(other, self.supported_agent_type)
                   })
       
       @property
       def key(self):
           return 'message'
       
       @property
       def supported_agent_type(self):
           return BroadcastingAgent
       
       def get_obs(self, agent, **kwargs):
           if not isinstance(agent, self.supported_agent_type):
               return {}
           
           obs = {other: 0 for other in agent.observation_space[self.key]}
           receive_from = self._broadcasting_state.update_message_and_reset_receiving(agent)
           for agent_id, message in receive_from:
               obs[agent_id] = message
           obs[agent.id] = agent.message
           return obs

Finally, we can create a custom done condition. We want the broadcasting agents to
finish when they've reached consensus; that is, when their internal message is within
some tolerance of the average message.

.. code-block:: python



