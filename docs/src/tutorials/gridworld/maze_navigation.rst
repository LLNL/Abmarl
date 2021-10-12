
.. Abmarl documentation GridWorld Maze Navigation tutorial.

Maze Navigation
---------------

Using the same GridWorld components as the :ref:`team battle tutorial <gridworld_tutorial_team_battle>`,
we can create a Maze Navigation simulation.
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
