.. Abmarl documentation installation instructions.

.. _installation:

Installation
============

User Installation
-----------------
You can install abmarl via `pip`:

.. code-block::

   pip install abmarl


Developer Installation
----------------------
To install Abmarl for development, first clone the repository and then install
via pip's development mode.

.. code-block::

   git clone git@github.com:LLNL/Abmarl.git
   cd abmarl
   pip install -r requirements.txt
   pip install -e . --no-deps


.. WARNING::
   If you are using `conda` to manage your virtual environment, then you must also
   install ffmpeg.


Dependency Note
---------------
Abmarl has the following dependencies

* Python 3.7+
* Tensorflow 2.4+
* Ray 1.2.0
* matplotlib
* seaborn
