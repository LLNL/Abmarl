
.. Abmarl documentation GridWorld communication tutorial.

.. _tutorials_gridworld_communication:

Communication Blocking
======================

Consider a simulation in which some agents send messages to each
other in an attempt to reach consensus while another group of agents attempts to
block these messages to impede consensus. Abmarl's GridWorld Simulation Framework
already contains the features for the blocking agents; in this tutorial, we show
how to create *new* components for the communication feature and connect them with
the simulation framework. The tutorial can be found in full
`in our repo <https://github.com/LLNL/Abmarl/blob/main/abmarl/examples/sim/comms_blocking.py>`_.

.. figure:: /.images/gridworld_tutorial_communications.*
   :width: 75 %
   :alt: Video showing agents attempting to block communication.

   Blockers (black) move around the maze blocking communications between broadcasters (green).
   The simulation ends when the broadcasters reach consensus.


Using built-in features
```````````````````````

Let's start by laying the groundwork using components already in Abmarl. We
create a simulation with :ref:`position <gridworld_position>`,
:ref:`movement <gridworld_movement>`, and :ref:`observations <gridworld_single_observer>`.

.. code-block:: python

   from matplotlib import pyplot as plt
   import numpy as np
   
   from abmarl.sim.gridworld.agent import MovingAgent, GridObservingAgent
   from abmarl.sim.gridworld.base import GridWorldSimulation
   from abmarl.sim.gridworld.state import PositionState
   from abmarl.sim.gridworld.actor import MoveActor
   from abmarl.sim.gridworld.observer import SingleGridObserver

   class BlockingAgent(MovingAgent, GridObservingAgent):
       def __init__(self, **kwargs):
           super().__init__(blocking=True, **kwargs)

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


Creating our own communication components
`````````````````````````````````````````

Next we build the communication components ourselves. We know that the GridWorld
Simulation Framework is made up of :ref:`Agents <gridworld_agent>`, :ref:`States <gridworld_state>`,
:ref:`Actors <gridworld_actor>`, :ref:`Observers <gridworld_observer>`, and
:ref:`Dones <gridworld_done>`, so we expect that we'll need to create each of these
for our new communication feature. Let's start with the Agent component.

An agent communicates by broadcasting its message to other nearby agents.
So we create a new agent with a `broadcast range` and an `initial message`. The
`broadcast range` will be used by the BroadcastActor to determine successful broadcasting,
and the `initial message`, an optional parameter, will be used by the BroadcastState
to set its message.

.. code-block:: python

   from abmarl.sim import Agent
   from abmarl.sim.gridworld.agent import GridWorldAgent

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

.. NOTE::

   We could have split the BroadcastingAgent into two agents types: one type of agent that has an
   internal message and another type that broadcasts. This is usually a better
   approach because it allows you to separate features and use them in greater
   combination with other features. We put them together in this tutorial for simplicity.

Next, we create the BroadcastState. This component manages the part of the simulation
state that tracks which messages have been sent among the agents. It will be used
by the BroadcastObserver to create the agent's observations. It also manages updates
to each agent's message.

.. code-block:: python

   from abmarl.sim.gridworld.state import StateBaseComponent

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
           """
           Update messages received from other agents.
           """
           for agent in to_agents:
               self.receiving_state[agent.id].append((from_agent.id, from_agent.message))
   
       def update_message_and_reset_receiving(self, agent):
           """
           Update agent's internal message.

           The agent averages all the messages that it has received from other
           agents in this step.
           """
           receiving_from = self.receiving_state[agent.id]
           self.receiving_state[agent.id] = []
   
           messages = [message for _, message in receiving_from]
           messages.append(agent.message)
           agent.message = np.average(messages)
   
           return receiving_from

Then we define the BroadcastActor. Similar to the
:ref:`BinaryAttackActor <gridworld_binary_attack>`, broadcasting will be a
boolean action--either broadcast or don't broadcast. We provide a `broadcast mapping`
for determining to which encodings each agent can broadcast. The message will be
successfully sent to every agent that (1) is within the `broadcast range`, (2) has
a compatible encoding, and (3) is not blocked.

.. code-block:: python

   from gym.spaces import Discrete
   from abmarl.sim.gridworld.actor import ActorBaseComponent
   import abmarl.sim.gridworld.utils as gu
   
   class BroadcastingActor(ActorBaseComponent):
       """
       Process sending and receiving messages between agents.
   
       BroadcastingAgents can broadcast to compatible agents within their range
       according to the broadcast mapping and if the agent is not blocked.
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

Now we define the BroadcastObserver. The observer enables agents to see all received
messages, including their own current message. This observer is unique from all
other components we have seen so far because it explicitly relies on the BroadcastingState
component, which will have a small impact in how we initialize the simulation.

.. code-block:: python

   from gym.spaces import Dict, Box
   from abmarl.sim.gridworld.observer import ObserverBaseComponent

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

   from abmarl.sim.gridworld.done import DoneBaseComponent

   class AverageMessageDone(DoneBaseComponent):
       def __init__(self, done_tolerance=None, **kwargs):
           super().__init__(**kwargs)
           self.done_tolerance = done_tolerance
   
       @property
       def done_tolerance(self):
           return self._done_tolerance
       
       @done_tolerance.setter
       def done_tolerance(self, value):
           assert type(value) in [int, float], "Done tolerance must be a number."
           assert value > 0, "Done tolerance must be positive."
           self._done_tolerance = value
   
       def get_done(self, agent, **kwargs):
           if isinstance(agent, BroadcastingAgent):
               average = np.average([
                   other.message for other in self.agents.values()
                   if isinstance(other, BroadcastingAgent)
               ])
               return np.abs(agent.message - average) <= self.done_tolerance
           else:
               return False
       
       def get_all_done(self, **kwargs):
           for agent in self.agents.values():
               if isinstance(agent, BroadcastingAgent):
                   if not self.get_done(agent):
                       return False
           return True

Building and running the simulation
```````````````````````````````````

Now that all the components have been created, we can create the full simulation:

.. code-block:: python

   from abmarl.sim.gridworld.base import GridWorldSimulation

   class BroadcastSim(GridWorldSimulation):
       def __init__(self, **kwargs):
           self.agents = kwargs['agents']
   
           self.position_state = PositionState(**kwargs)
           self.broadcasting_state = BroadcastingState(**kwargs)
   
           self.move_actor = MoveActor(**kwargs)
           self.broadcast_actor = BroadcastingActor(**kwargs)
   
           self.grid_observer = SingleGridObserver(**kwargs)
           self.broadcast_observer = BroadcastObserver(broadcasting_state=self.broadcasting_state, **kwargs)
   
           self.done = AverageMessageDone(**kwargs)
   
           self.finalize()
   
       def reset(self, **kwargs):
           self.position_state.reset(**kwargs)
           self.broadcasting_state.reset(**kwargs)
   
           self.rewards = {agent.id: 0 for agent in self.agents.values()}
   
       def step(self, action_dict, **kwargs):
           # process broadcasts
           for agent_id, action in action_dict.items():
               agent = self.agents[agent_id]
               receiving_agents = self.broadcast_actor.process_action(agent, action, **kwargs)
               if receiving_agents is not None:
                   self.broadcasting_state.update_receipients(agent, receiving_agents)
   
           # process moves
           for agent_id, action in action_dict.items():
               agent = self.agents[agent_id]
               move_result = self.move_actor.process_action(agent, action, **kwargs)
               if not move_result:
                   self.rewards[agent.id] -= 0.1
   
           # Entropy penalty
           for agent_id in action_dict:
               self.rewards[agent_id] -= 0.01
       
       def render(self, **kwargs):
           super().render(**kwargs)
           for agent in self.agents.values():
               if isinstance(agent, BroadcastingAgent):
                   print(f"{agent.id}: {agent.message}")
           print()
       
       def get_obs(self, agent_id, **kwargs):
           agent = self.agents[agent_id]
           return {
               **self.grid_observer.get_obs(agent, **kwargs),
               **self.broadcast_observer.get_obs(agent, **kwargs)
           }
       
       def get_reward(self, agent_id, **kwargs):
           reward = self.rewards[agent_id]
           self.rewards[agent_id] = 0
           return reward
   
       def get_done(self, agent_id, **kwargs):
           return self.done.get_done(agent_id, **kwargs)
       
       def get_all_done(self, **kwargs):
           return self.done.get_all_done(**kwargs)
       
       def get_info(self, **kwargs):
           return {}
   
Let's initialize our simulation and run it. We initialize some BroadcastingAgents
and some BlockingAgents. Then we initialize the simulation with a `broadcast mapping`
that specifies that broadcasts can only be made amongst agents with encoding 1, which
are the BroadcastingAgents.

.. code-block:: python

   agents = {
       'broadcaster0': BroadcastingAgent(id='broadcaster0', encoding=1, broadcast_range=6, render_color='green'),
       'broadcaster1': BroadcastingAgent(id='broadcaster1', encoding=1, broadcast_range=6, render_color='green'),
       'broadcaster2': BroadcastingAgent(id='broadcaster2', encoding=1, broadcast_range=6, render_color='green'),
       'broadcaster3': BroadcastingAgent(id='broadcaster3', encoding=1, broadcast_range=6, render_color='green'),
       'blocker0': BlockingAgent(id='blocker0', encoding=2, move_range=2, view_range=3, render_color='black'),
       'blocker1': BlockingAgent(id='blocker1', encoding=2, move_range=1, view_range=3, render_color='black'),
       'blocker2': BlockingAgent(id='blocker2', encoding=2, move_range=1, view_range=3, render_color='black'),
   }
   sim = BroadcastSim.build_sim(
       7, 7,
       agents=agents,
       broadcast_mapping={1: [1]},
       done_tolerance=5e-10
   )
   
   sim.reset()
   fig = plt.figure()
   sim.render(fig=fig)
   
   done_agents = set()
   for i in range(50):
       action = {
           agent.id: agent.action_space.sample() for agent in agents.values() if agent.id not in done_agents
       }
       sim.step(action)
       for agent in agents:
           if agent not in done_agents:
               obs = sim.get_obs(agent)
           if sim.get_done(agent):
               done_agents.add(agent)
   
       sim.render(fig=fig)
       if sim.get_all_done():
           break

The visualization produces an animation like the one at the top of this page.
We can see the "path towards consensus" among the BroadcastingAgents in the output.
Keep your eye open for the effects of blocking.

.. code-block::

   Step 1
   broadcaster0: 0.5936447861764813
   broadcaster1: -0.8344218389696239
   broadcaster2: 0.09891331950679949
   broadcaster3: 0.32590416873488093
   
   Step 2
   broadcaster0: 0.028375705313912796
   broadcaster1: -0.25425883511737146
   broadcaster2: -0.13653478357598114
   broadcaster3: -0.25425883511737146
   
   For steps 3-5, notice that Broadcaster3 is blocked. The other broadcasters
   have reached a consensus, but the simulation does not end becaue they must all
   agree.
   
   Step 3
   broadcaster0: -0.12080597112647994
   broadcaster1: -0.12080597112647994
   broadcaster2: -0.12080597112647995
   broadcaster3: -0.15416918712420283
   
   Step 4
   broadcaster0: -0.12080597112647994
   broadcaster1: -0.12080597112647994
   broadcaster2: -0.12080597112647995
   broadcaster3: -0.15416918712420283
   
   Step 5
   broadcaster0: -0.12080597112647994
   broadcaster1: -0.12080597112647994
   broadcaster2: -0.12080597112647995
   broadcaster3: -0.15416918712420283
   
   Broadcaster3 is no longer blocked
   Step 6
   broadcaster0: -0.12080597112647995
   broadcaster1: -0.12080597112647995
   broadcaster2: -0.12080597112647995
   broadcaster3: -0.1319270431257209
   
   ...
   
   Step 16
   broadcaster0: -0.1241744002450772
   broadcaster1: -0.12417639653661512
   broadcaster2: -0.12417523451616769
   broadcaster3: -0.12417511533458334
   
   Step 17
   broadcaster0: -0.12417528665811084
   broadcaster1: -0.12417528665811083
   broadcaster2: -0.12417528665811083
   broadcaster3: -0.12417528665811084

Extra Challenges
----------------
Having successfully created new components and fit them into the GridWorld Simulation
Framework, we can create a vast variety of different simulations, constrained primarily
by our own imagination. We leave the extra challenges up to you and what you can
think of.

