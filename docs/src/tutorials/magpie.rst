.. Admiral documentation Magpie tutorial.

Magpie
======

# Running with magpie

The prospect of applying Reinforcement Learning algorithms in HPC
environments is very attractive. As a first step, we demonstrate that
admiral can be used with [magpie](https://github.com/LLNL/magpie) to create batch
jobs for running on multiple compute nodes.


## Installing Admiral on HPC systems

Here we'll use conda to install on an HPC system:
1. Create the conda virtual environment: `conda create --name admiral`
1. Activate it: `conda activate admiral`
1. Install pip installer: `conda install --name admiral pip`
1. Follow installation instructions on [main README](/README.md)

# Usage

We demonstrate running the [Predator-Prey example](/examples/predator_prey/predator_prey_training.py)
using Mapgie.

## make-runnable mode

Admiral's command line interface provides the `make-runnable`
subcommand that converts the configuration script into a runnable script and saves it
to the same directory.

```
admiral make-runnable predator_prey_training.py
```

This will create a file called `runnable_predator_prey_training.py`, which can be
run directly from the command line: This has the same effect as if you were using
admiral's train command.

## Magpie flag

The full use of `make-runnable` is seen when it is run with the `--magpie` flag.
This will create a custom magpie script using
[magpie's ray default script](https://github.com/LLNL/magpie/blob/master/submission-scripts/script-sbatch-srun/magpie.sbatch-srun-ray)
as a starting point. This also adds the correct initialization parameters to
`ray.init()` in the `runnable_` script. For example,

```
admiral make-runnable predator_prey_training.py --magpie
```

will create the `runnable_` script with `ray.init(address=os.environ['MAGPIE_RAY_ADDRESS'])`
and will create a [magpie batch script](/examples/predator_prey/PredatorPrey_magpie.sbatch-srun-ray)
that is setup to run this example. To launch the batch job, we simply run it from the command line:

```
sbatch -k --ip-isolate=yes PredatorPrey_magpie.sbatch-srun-ray
```

The script can be modified to adjust the job parameters, such as the number of
compute nodes, the time limit for the job, etc. This can also be done through
admiral via the `-n` and `-t` options. Note: the `num_workers` parameter in the
tune configuration is the number of processors to utilize per compute node, which
is the different from the number of compute nodes you are requesting.




