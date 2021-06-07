# Abmarl

Abmarl is a package for developing agent-based simulations and training them with
multiagent reinforcement learning. We provide an intuitive command line interface
for training, visualizing, and analyzing agent behavior. We define an Agent Based
Simulation Interface and Simulation Managers, which control which agents interact
with the simulation at each step. We support integration with several popular simulation
interfaces, including gym.Env and MultiAgentEnv.

Abmarl is a layer in the Reinforcement Learning stack that sits on top of RLlib.
We leverage RLlib’s framework for training agents and extend it to more easily support
custom simulations, algorithms, and policies. We enable researchers to rapidly
prototype RL experiments and simulation design and lower the barrier for pre-existing
projects to prototype RL as a potential solution.

<p align="center">
  <img src="https://github.com/LLNL/Abmarl/actions/workflows/build-and-test.yml/badge.svg" alt="Build and Test Badge" />
  <img src="https://github.com/LLNL/Abmarl/actions/workflows/build-docs.yml/badge.svg" alt="Sphinx docs Badge" />
  <img src="https://github.com/LLNL/Abmarl/actions/workflows/lint.yml/badge.svg" alt="Lint Badge" />
</p>


## Getting started

To use Abmarl, install via pip: `pip install abmarl`

To develop Abmarl, clone the repository and install via pip's development mode:

```
git clone git@github.com:LLNL/Abmarl.git
cd abmarl
pip install -r requirements.txt
pip install -e . --no-deps
```

Train agents in a multicorridor simulation:
```
abmarl train examples/multi_corridor_example.py
```

## Documentation

You can find the latest Abmarl documentation, on
[our ReadTheDocs page](https://abmarl.readthedocs.io/en/latest/index.html).

[![Documentation Status](https://readthedocs.org/projects/abmarl/badge/?version=latest)](https://abmarl.readthedocs.io/en/latest/?badge=latest)


## Contact

* Edward Rusu, rusu1@llnl.gov
* Ruben Glatt, glatt1@llnl.gov

## Release

LLNL-CODE-815883

