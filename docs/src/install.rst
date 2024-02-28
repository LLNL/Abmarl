.. Abmarl documentation installation instructions.

.. _installation:

Installation
============

Installation supports the core functionality of abmarl, such as the
:ref:`agent based simuluation <abs>` and the :ref:`GridWorldSimluation Framework <gridworld>`,
and the :ref:`external functionality <external>` with support for packages like RLlib.

User Installation
-----------------
You can install abmarl via `pip`. Specifying `rllib` or `open-spiel` will install
the dependencies necessary for using those packages with Abmarl. If neither is specified,
then only the core Abmarl features will be installed.

Install just Abmarl's core functionality with

.. code-block::

   pip install abmarl

Add extra packages for integration with RLlib or Open Spiel with

.. code-block::

   pip install abmarl[rllib]
   pip install abmarl[open-spiel]

Developer Installation
----------------------
To install Abmarl for development, first clone the repository and then install
via pip's development mode.

.. code-block::

   git clone git@github.com:LLNL/Abmarl.git
   cd abmarl
   pip install -r requirements/requirements_all.txt
   pip install -e . --no-deps

You can pick among ``requirements_{all, core, dev, workflow}.txt`` when installing
dependencies.

.. WARNING::
   If you are using `conda` to manage your virtual environment, then you must also
   install ffmpeg.
