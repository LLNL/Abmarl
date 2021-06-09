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
via pip's development mode. Note: Abmarl requires `python3.7+`.

.. code-block::

   git clone git@github.com:LLNL/Abmarl.git
   cd abmarl
   pip install -r requirements.txt
   pip install -e . --no-deps
