.. Admiral documentation index

Welcome to Admiral's documentation!
===================================

Admiral is a package for developing agent-based simulations and training them
with multiagent reinforcement learning. We provide an intuitive command line
interface for training, visualizing, and analyzing agent behavior. We define an
:ref:`Agent Based Simulation Interface <abs>` and :ref:`Simulation Managers <sim-man>`,
which control which agents interact with the simulation at each step. We support
:ref:`integration <external>` with several popular simulation interfaces, including
:ref:`gym.Env <api_gym_wrapper>` and :ref:`MultiAgentEnv <api_ma_wrapper>`.

Admiral is a layer in the Reinforcement Learning stack that sits on top of RLlib.
We leverage RLlib's framework for training agents and extend it to more easily
support custom simulations, algorithms, and policies. We enable researchers to
rapidly prototype RL experiments and simulation design and lower the barrier
for pre-existing projects to prototype RL as a potential solution.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   highlights
   install
   tutorials/tutorials
   api
