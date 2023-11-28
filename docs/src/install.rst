.. Abmarl documentation installation instructions.

.. _installation:

Installation
============

User Installation
-----------------
You can install abmarl via `pip`:

.. code-block::

   pip install abmarl[, rllib, open-spiel]

Specifying `rllib` or `open-spiel` will install the dependencies necessary for
using those packages with Abmarl. If neither is specified, then the core Abmarl
features will be installed, such as the :ref:`AgentBasedSimulation <abs>` interface
and the :ref:`GridWorldSimluation Framework <gridworld>`.


Developer Installation
----------------------
To install Abmarl for development, first clone the repository and then install
via pip's development mode.

.. code-block::

   git clone git@github.com:LLNL/Abmarl.git
   cd abmarl
   pip install -r requirements/requirements_all.txt
   pip install -e . --no-deps


.. WARNING::
   If you are using `conda` to manage your virtual environment, then you must also
   install ffmpeg.
