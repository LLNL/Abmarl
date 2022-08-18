
.. Abmarl documentation GridWorld Team Battle tutorial.

.. _gridworld_tutorial_team_battle:

Team Battle
===========

The Team Battle scenario involves multiple teams of agents fighting against each other.
The goal of each team is to be the last team alive, at which point the simulation will end.
Each agent can move around the grid and attack agents from other teams. Each agent
can observe the grid around its position. We will reward each agent for successful
kills and penalize them for bad moves. This tutorial can be found in full
`in our repo <https://github.com/LLNL/Abmarl/blob/main/abmarl/examples/sim/team_battle_example.py>`_.

.. figure:: /.images/gridworld_tutorial_team_battle.*
   :width: 75 %
   :alt: Video showing teams fighting.

   Agents on four teams battling each other.

First, we import the components that we need. Each component is
:ref:`already in Abmarl <gridworld_built_in_features>`, so we don't need to create anything new.

.. code-block:: python

   from matplotlib import pyplot as plt
   import numpy as np

   from abmarl.sim.gridworld.base import GridWorldSimulation
   from abmarl.sim.gridworld.agent import GridObservingAgent, MovingAgent, AttackingAgent, HealthAgent
   from abmarl.sim.gridworld.state import HealthState, PositionState
   from abmarl.sim.gridworld.actor import MoveActor, BinaryAttackActor
   from abmarl.sim.gridworld.observer import SingleGridObserver
   from abmarl.sim.gridworld.done import OneTeamRemainingDone

Then, we define our agent types. This simulation will only have a single type:
the BattleAgent. Most of the agents' attributes will be the same, and we can preconfigure
the class definition to save us time when we create the agents later on.

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
simulation.

.. code-block:: python

   class TeamBattleSim(GridWorldSimulation):
       def __init__(self, **kwargs):
           self.agents = kwargs['agents']
   
           # State Components
           self.position_state = PositionState(**kwargs)
           self.health_state = HealthState(**kwargs)
   
           # Action Components
           self.move_actor = MoveActor(**kwargs)
           self.attack_actor = BinaryAttackActor(**kwargs)
   
           # Observation Components
           self.grid_observer = SingleGridObserver(**kwargs)
   
           # Done Compoennts
           self.done = OneTeamRemainingDone(**kwargs)
           
           self.finalize()

Next we define the start state of each simulation. We lean on the
:ref:`State Components <gridworld_state>` to perform the reset. Note that we
must track the rewards explicitly.

.. code-block:: python

   class TeamBattleSim(GridWorldSimulation):
       ...
 
       def reset(self, **kwargs):
           self.position_state.reset(**kwargs)
           self.health_state.reset(**kwargs)
        
           # Track the rewards
           self.rewards = {agent.id: 0 for agent in self.agents.values()}

Then we define how the simulation will step forward, leaning on the :ref:`Actors <gridworld_actor>`
to process their part of the action. The Actors' result determine the agents'
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

Finally, we define each of the getters using the :ref:`Observers <gridworld_observer>`
and :ref:`Done components <gridworld_done>`.

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

Now that we've defined our agents and simulation, let's create them and run it.
First, we'll create the agents. There will be 4 teams, so we want to
color the agent by team and start them at different corners of the grid. Besides that,
all agent attributes will be the same, and here we benefit from preconfiguring
the attributes in the class definition above.

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

Finally, we can run the simulation with random actions and visualize it. The visualization
produces an animation like the one at the top of this page.

.. code-block:: python

   sim.reset()
   fig = plt.figure()
   sim.render(fig=fig)

   done_agents = set()
   for i in range(50): # Run for at most 50 steps
       action = {
           agent.id: agent.action_space.sample() for agent in agents.values() if agent.id not in done_agents
       }
       sim.step(action)
       sim.render(fig=fig)
   
       if sim.get_all_done():
           break
       for agent in agents:
           if sim.get_done(agent):
               done_agents.add(agent)

Extra Challenges
````````````````
Having successfully created and run a TeamBattle simulation, we can further explore
the GridWorldSimulation framework. Some ideas are:

* Experiment with the number of agents and the impact that has on both the SingleGridObserver
  and the MultiGridObserver.
* Experiment with the number of agents per team as well as the capabilities of
  those agents. You might find that a super capable agent is still effective against
  a team of multiple agents.
* Create a Hunter-Forager simulation, where one team of agents act as immobile resources
  that can be foraged by another team, which can be hunted by a third team. Try
  using the same components here, although you may need to use a custom
  :ref:`done condition <gridworld_done>`.
* Connect this simulation with the Reinforcement Learning capabilities of Abmarl
  via a :ref:`Simulation Manager<sim-man>`. What kind of behaviors do the agents
  learn?
* And much, much more!