.. Abmarl latest releases.

What's New in Abmarl
====================

Abmarl version 0.2.8 features
a new :ref:`use case <jcats_nav>`, showcasing Abmarl's
usage as a proxy simulator to inform reinforcement learning training in an external
simulator;
a refactored cli scripting interface, allowing each Abmarl command (debug, train,
visualize, an analyze) to be issued :ref:`in a python script <commands_in_python_script>`
in addition to being run from the terminal;
and an :ref:`OrientationAgent <gridworld_orientation_drifting>` and
:ref:`DriftMoveActor <gridworld_orientation_drifting>` support moving an agent through
the grid by drifting it in the direction it is facing.


Using Abmarl as a Proxy Simulation
----------------------------------

Abmarl's :ref:`GridWorld Simulation Framework <gridworld>` has shown promise as
a proxy simulation to iterate the training experience in a reinforcement learning
experiment using an external simulator. In this experiment, researchers used Abmarl's
:ref:`simulation interface <sim-man>` to connect a C++ based conflict simulation
JCATS to reinforcement learning algorithms in order to train an agent to navigate
to a waypoint. All state updates are controlled by the JCATS simulation itself.
Positional observations are reported to the RL policy, which in turn issues movement
commands to the the simulator. Researchers leveraged Abmarl as a proxy simulation
to rapidly find a warm start configuration. Training is performed on a cluster of
4 nodes utilizing RLlib's client-server architecture. They successfully generated
136 million training steps and trained the agent to navigate the scenario. See
:ref:`the featured description <jcats_nav>` for more information.


Abmarl Commands in a Python Script
----------------------------------

All of Abmarl's CLI commands can be :ref:`used directly in a python script <commands_in_python_script>`
instead of relying on the CLI by importing those moodules and running them with
the experiment configuration. See :ref:`Python Scripts <commands_in_python_script>`
and `a full workflow example <https://github.com/LLNL/Abmarl/blob/main/examples/full_workflow.py>`_
for more details.


Drifting Agents
---------------
The combination of the :ref:`OrientationAgent <gridworld_orientation_drifting>` and
:ref:`DriftMoveActor <gridworld_orientation_drifting>` allows us to move an agent
through the grid by drifting it in the direction it is facing. An agent can attempt
to change its direction, and the Actor will attempt to move it in whatever direction
it is facing.


Miscellaneous
-------------

Interface changes
`````````````````

* Exploration is off by default during visualization and can be turned on with the
  ``--explore`` flag.
* :ref:`External wrappers <external>` now support the ``unwrapped`` property, which
  returns the underlying :ref:`Simulation Manager <sim-man>` object.

Other Features
``````````````

* Support for :ref:`installing Abmarl <installation>` with various extras: core,
  develop, workflow, and all.
* :ref:`Ranomize placement order <api_gridworld_state_position>` parameter for randomly
  iterating through dictionary of agents when initially placing them in the grid.
* Gridworld agents have customizable ``render_size``, which defaults to 200.
* Traffic corridor simulates agents navigating
  a tight corridor. They must cooperate by taking turns in order for all of them
  to make it through.
* Pacman variation simulates the Pacman arcade game,
  with support for training the "baddie" agents too.

Bug fixes
`````````

* :ref:`Ravel <ravel_wrapper>` and :ref:`flatten <flatten_wrapper>` support Gym Boxes.
* :ref:`AttackActors <gridworld_attacking>` check to see if the attackable agent
  has health.
