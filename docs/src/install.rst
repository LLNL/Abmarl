.. Admiral documentation installation instructions.

Installation
============

Simple Installation
-------------------
Install from requirements file. This uses tensorflow and installs all you need
to run the examples.
1. Install the requirements: `pip install -r requirements.txt`
1. Install Admiral `pip install .` or `pip install -e .`


Detailed Installation
---------------------
Install each package as needed.

To train:
1. Install tensorflow or pytorch
1. Install ray rllib v1.2.0: `pip install ray[rllib]==1.2.0`
1. Install Admiral: `pip install .` or `pip install -e .`

To visualize:
1. Install matplotlib: `pip install matplotlib`

To run Predator-Prey example:
1. Install seaborn: `pip install seaborn`

