.. Abmarl documentation dssi rl class installation instructions.

.. _dssi_rl_setup:

DSSI RL Setup
=============

Unix Users
----------

If you have Mac or Linux, you can install ABMARL directly onto your machine.

First, download the repository:

.. code_block::

   git clone https://github.com/LLNL/Abmarl.git
   cd Abmarl
   git checkout abmarl-dssi-class


Then, create a virtual environment and install ABMARL from the source

.. code_block::

   python3 -m venv v_abmarl
   source v_abmarl/bin/activate
   pip3 install --upgrade pip3
   pip3 install -r requirements.txt
   pip3 install -e . --no-deps

Test that your installation is successful

.. code_block::

   abmarl train examples/multi_corridor_example.py

This will create `abmarl_results` in your home directory and will store the trained
behavior in a subdirectory with the name of the simulation, `MultiCorridor`, followed
by the date and time (e.g. `MultiCorridor_2021-07-20_17-36`). You can confirm the
trained behavior with

.. code_block::

   abmarl visualize ~/abmarl_results/MultiCorridor_2021-07-20_17-36/

You should see five green squares moving to the right.


Windows/Amazon Workspace Users
------------------------------

If you have Windows and/or are using the Amazon Workspace, you can connect to LC
via VNC and use the ABMARL installation on LC. You must have an LC account and
`VNC viewer <https://hpc.llnl.gov/software/visualization-software/vnc-realvnc>`_
installed on your system.

Once you've done that, ssh onto the system of choice (probably flash for
DSSI students).

.. code_block

   ssh -X -Y username@flash.llnl.gov

Then, download the repository:

.. code_block::

   git clone https://github.com/LLNL/Abmarl.git
   cd Abmarl
   git checkout abmarl-dssi-class

Then source the virtual environment

.. code_block::

   source /usr/workspace/DSSI-RL/dssi-rl/bin/activate

Test that your installation is successful

.. code_block::

   abmarl train examples/multi_corridor_example.py

This will create `abmarl_results` in your home directory and will store the trained
behavior in a subdirectory with the name of the simulation, `MultiCorridor`, followed
by the date and time (e.g. `MultiCorridor_2021-07-20_17-36`). You can confirm the
trained behavior with

.. code_block::

   abmarl visualize ~/abmarl_results/MultiCorridor_2021-07-20_17-36/

You should see five green squares moving to the right.


.. WARNING::
   If you are using `conda` to manage your virtual environment, then you must also
   install ffmpeg.
