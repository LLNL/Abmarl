.. Abmarl documentation index

Welcome to Abmarl's documentation!
===================================

Abmarl is a package for developing Agent-Based Simulations and training them
with MultiAgent Reinforcement Learning (MARL). We provide an intuitive command line
interface for engaging with the full workflow of MARL experimentation: training,
visualizing, and analyzing agent behavior. We define an
:ref:`Agent-Based Simulation Interface <abs>` and :ref:`Simulation Manager <sim-man>`,
which control which agents interact with the simulation at each step. We support
:ref:`integration <external>` with popular reinforcement learning simulation interfaces, including
:ref:`gym.Env <api_gym_wrapper>` and :ref:`MultiAgentEnv <api_ma_wrapper>`. We
define our own :ref:`GridWorld Simulation Framework <gridworld>` for creating custom grid-based
Agent Based Simulations.

Abmarl leverages RLlib's framework for reinforcement learning and extends it to more easily
support custom simulations, algorithms, and policies. We enable researchers to
rapidly prototype MARL experiments and simulation design and lower the barrier
for pre-existing projects to prototype RL as a potential solution.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   gridworld
   featured_usage
   install
   tutorials/tutorials
   api
