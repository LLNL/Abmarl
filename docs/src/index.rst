.. Admiral documentation index

Welcome to Admiral's documentation!
===================================

Admiral is a package for developing agent based simulations and training them
with multiagent reinforcement learning. We provide an intuitive command line
interface for training, visualizing, and analyzing agent behavior. We define an
:ref:`Agent Based Simulation Interface <abs>` and :ref:`Simulation Managers <sim-man>`,
which control which agents interact
with the environment at each step. We support :ref:`integration <external>` with several popular
environment interfaces, including `gym.Env <https://github.com/openai/gym/blob/master/gym/core.py#L8>`_
and `MultiAgentEnv <https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py#L12>`_.

.. ATTENTION::
   TODO: can we link to pages in the API output? If so, replace the external links
   with internal links to the API output.

Admiral is a layer in the Reinforcement Learning stack that sits on top of RLlib.
We leverage RLlib's framework for training agents and extend it to more easily
support custom environments, algorithms, and policies. We enable researchers to
rapidly prototype RL experiments and environment design and lower the barrier
for pre-existing projects to prototype RL as a potential solution.

.. toctree::
   :maxdepth: 2
   :caption: Site Map

   overview
   highlights
   install
   tutorials/tutorials
   api
