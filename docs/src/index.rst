.. Admiral documentation index

Welcome to Admiral's documentation!
===================================

Admiral is a package for developing agent based simulations and training them
with multiagent reinforcement learning. We provide an intuitive command line
interface for training, visualizing, and analyzing agent behavior. We define an
[Agent Based Simulation interface](/admiral/envs/agent_based_simulation.py) and
[Simulation Managers](/admiral/managers/), which control which agents interact
with the environment at each step. We support integration with several popular
environment interfaces, including [gym.Env](/admiral/external/gym_env_wrapper.py) and
[MultiAgentEnv](/admiral/external/rllib_multiagentenv_wrapper.py).

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
