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
:ref:`gym.Env <gym_external>`, :ref:`MultiAgentEnv <rllib_external>`, and
:ref:`OpenSpiel <open_spiel_external>`. We define our own
:ref:`GridWorld Simulation Framework <gridworld>` for creating custom grid-based
Agent Based Simulations.

Abmarl leverages RLlib's framework for reinforcement learning and extends it to more easily
support custom simulations, algorithms, and policies. We enable researchers to
rapidly prototype MARL experiments and simulation design and lower the barrier
for pre-existing projects to prototype RL as a potential solution.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   release
   overview
   gridworld
   featured_usage/featured_usage
   install
   tutorials/tutorials


Citation
--------

Abmarl has been `published in the Journal of Open Source Software <https://joss.theoj.org/papers/10.21105/joss.03424>`_.
It can be cited using the following bibtex entry:

.. code-block::

   @article{Rusu2021,
     doi = {10.21105/joss.03424},
     url = {https://doi.org/10.21105/joss.03424},
     year = {2021},
     publisher = {The Open Journal},
     volume = {6},
     number = {64},
     pages = {3424},
     author = {Edward Rusu and Ruben Glatt},
     title = {Abmarl: Connecting Agent-Based Simulations with Multi-Agent Reinforcement Learning},
     journal = {Journal of Open Source Software}
   }
