.. Admiral documentation installation instructions.

.. _installation:

Installation
============

Simple Installation
-------------------
Install from the requirements file. This uses tensorflow and installs all you need
to run the examples.

* Install the requirements: ``pip install -r requirements.txt``
* Install Admiral: ``pip install .`` or ``pip install -e .``


Detailed Installation
---------------------
Install each package as needed.

For Training
````````````

* Install tensorflow or pytorch
* Install ray rllib v1.2.0: ``pip install ray[rllib]==1.2.0``
* Install Admiral: ``pip install .`` or ``pip install -e .``

For Visualizing
```````````````

* Install matplotlib: ``pip install matplotlib``

For the Tutorials
`````````````````

* Install seaborn: ``pip install seaborn``

