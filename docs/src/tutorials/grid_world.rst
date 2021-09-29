
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
kills and penalize them for bad moves. We will demonstrate configuring the agents
homogeneously and heterogeneously.

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
the BattleAgent. Furthermore, the agent will not be pre-configured with any parameters,
so definiing the BattleAgent is just boilerplate.

.. code-block:: python

   class BattleAgent(GridObservingAgent, MovingAgent, AttackingAgent, HealthAgent):
       pass

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
their part of the action.

.. code-block:: python

   class TeamBattleSim(GridWorldSimulation):
       ...

       def step(self, action_dict, **kwargs):
           # Process attacks:
           for agent_id, action in action_dict.items():
               agent = self.agents[agent_id]
               if agent.active:
                   self.attack_actor.process_action(agent, action, **kwargs)
 
           # Process moves
           for agent_id, action in action_dict.items():
               agent = self.agents[agent_id]
               if agent.active:
                   self.move_actor.process_action(agent, action, **kwargs)

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
           mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, edgecolor='black', facecolor='gray')
   
           plt.plot()
           plt.pause(1e-6)
