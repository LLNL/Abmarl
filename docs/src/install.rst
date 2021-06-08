.. Admiral documentation installation instructions.

.. _installation:

Installation
============

User Installation
-----------------
You can install admiral via `pip`:

.. code-block::

   pip install abmarl


Developer Installation
----------------------
To install Admiral for development, first clone the repository and then install
via pip's development mode:

.. code-block::

   git clone git@github.com:LLNL/Admiral.git
   cd admiral
   pip install -r requirements.txt
   pip install -e . --no-deps
