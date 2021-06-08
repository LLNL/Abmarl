# Admiral

Admiral is a package for developing Agent-Based Simulations and training them with
MultiAgent Reinforcement Learning (MARL). We provide an intuitive command line
interface for engaging with the full workflow of MARL experimentation: training,
visualizing, and analyzing agent behavior. We define an Agent-Based
Simulation Interface and Simulation Manager, which control which agents interact
with the simulation at each step. We support integration with popular reinforcement
learning simulation interfaces, including gym.Env and MultiAgentEnv.

Admiral leverages RLlib’s framework for reinforcement learning and extends it to
more easily support custom simulations, algorithms, and policies. We enable researchers to rapidly
prototype MARL experiments and simulation design and lower the barrier for pre-existing
projects to prototype RL as a potential solution.

<p align="center">
  <img src="https://github.com/LLNL/Admiral/actions/workflows/build-and-test.yml/badge.svg" alt="Build and Test Badge" />
  <img src="https://github.com/LLNL/Admiral/actions/workflows/build-docs.yml/badge.svg" alt="Sphinx docs Badge" />
  <img src="https://github.com/LLNL/Admiral/actions/workflows/lint.yml/badge.svg" alt="Lint Badge" />
</p>


## Quickstart

* Clone the repository
* Install the requirements: `pip install -r requirements.txt`
* Install Admiral: `pip install .` or `pip install -e .`

Train agents in a multicorridor simulation:
```
admiral train examples/multi_corridor_example.py
```

## Documentation

You can find the latest Admiral documentation, on
[our ReadTheDocs page](https://abmarl.readthedocs.io/en/latest/index.html).

[![Documentation Status](https://readthedocs.org/projects/abmarl/badge/?version=latest)](https://abmarl.readthedocs.io/en/latest/?badge=latest)


## Community

### Reporting Issues

Please use our issue tracker to report any issues or submit feature requests. Great
bug reports tend to have:
- A quick summary and/or background
- Steps to reproduce, sample code is best.
- What you expected would happen
- What actually happens

### Contributing

Please submit any changes via pull requests from a forked repository. Find out
more about this process [here](https://guides.github.com/introduction/flow/index.html).
All contributions are under the BSD 3 License that covers the project.

### Additional support

* Edward Rusu, rusu1@llnl.gov
* Ruben Glatt, glatt1@llnl.gov

## Release

LLNL-CODE-815883
