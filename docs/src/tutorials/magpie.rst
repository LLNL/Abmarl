.. Abmarl documentation Magpie tutorial.

.. _tutorial_magpie:

Magpie
======

The prospect of applying MuliAgent Reinforcement Learning algorithms on HPC
systems is very attractive. As a first step, we demonstrate that
abmarl can be used with `magpie <https://github.com/LLNL/magpie>`_ to create batch
jobs for running on multiple compute nodes.


Installing Abmarl on HPC systems
---------------------------------

Here we'll use conda to install on an HPC system:

* Create the conda virtual environment: `conda create --name abmarl`
* Activate it: `conda activate abmarl`
* Install pip installer: `conda install --name abmarl pip`
* Follow :ref:`installation instructions <installation>`

Usage
-----

We demonstrate running the :ref:`PredatorPrey tutorial <tutorial_predator_prey>`
using Mapgie.

make-runnable
`````````````
Abmarl's command line interface provides the `make-runnable`
subcommand that converts the configuration script into a runnable script and saves it
to the same directory.

.. code-block::

   abmarl make-runnable predator_prey_training.py

This will create a file called `runnable_predator_prey_training.py`.

Magpie flag
```````````
The full use of `make-runnable` is seen when it is run with the ``--magpie`` flag.
This will create a custom magpie script using
`magpie's ray default script <https://github.com/LLNL/magpie/blob/master/submission-scripts/script-sbatch-srun/magpie.sbatch-srun-ray>`_
as a starting point. This also adds the correct initialization parameters to
`ray.init()` in the `runnable_` script. For example,

.. code-block::

   abmarl make-runnable predator_prey_training.py --magpie


will create the `runnable_` script with ``ray.init(address=os.environ['MAGPIE_RAY_ADDRESS'])``
and will create a
`magpie batch script <https://github.com/LLNL/Abmarl/blob/main/examples/predator_prey/PredatorPrey_magpie.sbatch-srun-ray>`_
that is setup to run this example. To launch the batch job, we simply run it from
the command line:

.. code-block::

   sbatch -k --ip-isolate=yes PredatorPrey_magpie.sbatch-srun-ray

The script can be modified to adjust the job parameters, such as the number of
compute nodes, the time limit for the job, etc. This can also be done through
abmarl via the ``-n`` and ``-t`` options.

.. ATTENTION::
   the `num_workers` parameter in the tune configuration is the number of processors
   to utilize per compute node, which is the different from the number of compute
   nodes you are requesting.
