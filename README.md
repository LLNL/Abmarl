# Admiral

Admiral is a package for developing agent-based simulations and training them with
multiagent reinforcement learning. We provide an intuitive command line interface
for training, visualizing, and analyzing agent behavior. We define an Agent Based
Simulation Interface and Simulation Managers, which control which agents interact
with the environment at each step. We support integration with several popular environment
interfaces, including gym.Env and MultiAgentEnv.

Admiral is a layer in the Reinforcement Learning stack that sits on top of RLlib.
We leverage RLlib’s framework for training agents and extend it to more easily support
custom environments, algorithms, and policies. We enable researchers to rapidly
prototype RL experiments and environment design and lower the barrier for pre-existing
projects to prototype RL as a potential solution.

BADGES HERE OR ABOVE

## Getting started

* Clone the repository
* Install the requirements: `pip install -r requirements.txt`
* Install Admiral: `pip install .` or `pip install -e .`

Train a multicorridor environment:
```
admiral train examples/multi_corridor_example.py
```

## Documentation

You can find the latest Admiral documentation, on
[our ReadTheDocs page](https://abmarl.readthedocs.io/en/latest/index.html).


## Contact

* Edward Rusu, rusu1@llnl.gov
* Ruben Glatt, glatt1@llnl.gov

## Release

LLNL-CODE-815883

