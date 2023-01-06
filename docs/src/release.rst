.. Abmarl latest releases.

What's New in Abmarl
====================

Abmarl version 0.2.5 features a new set of built-in :ref:`Attack Actors <gridworld_attacking>`
for the :ref:`Grid World Simulation Framework <gridworld>`; an
:ref:`exclusive channel action wrapper <gridworld_exclusive_channel_action_wrapper>`
to allow users to isolate actions by channel; and an enahanced
:ref:`FlattenWrapper <flatten_wrapper>` for dealing with Discrete spaces.


New Attack Actors
-----------------

Abmarl's :ref:`Grid World Simulation Framework <gridworld>` now provides four built-in
:ref:`Attack Actors <gridworld_attacking>` for greater flexibility in processing
attacks. In addition to the already-supported :ref:`BinaryAttackActor <gridworld_binary_attack>`,
Abmarl now supports an :ref:`EncodingBasedAttackActor <gridworld_encoding_based_attack>`,
a :ref:`SelectiveAttackActor <gridworld_selective_attack>`, and a
:ref:`RestrictedSelectiveAttackActor <gridworld_restricted_selective_attack>`.
These attackors allow agents to specify attacks based on encodings and cells.
:ref:`AttackingAgents <api_gridworld_agent_attack>` can now use multiple
attacks per turn, up to their ``attack_count``, as interpreted by the Attack Actors.

In addition, :ref:`Attack Actors <gridworld_attacking>` now output the attack status
along with a list of attacked agents, allowing simulations to distinguish and issue
rewards among three outcomes:

   #. No attack attempted
   #. Attack attempted and failed
   #. Attack attempted and successful


Exclusive Channel Action Wrapper
--------------------------------

Users can enforce exclusivity of actions among the channels within a single Actor,
so that the agent must pick from one channel or the other. For example, the
:ref:`ExclusiveChannelActionWrapper <gridworld_exclusive_channel_action_wrapper>`
can be used with the :ref:`EncodingBasedAttackActor <gridworld_encoding_based_attack>`
for :ref:`AttackingAgents <api_gridworld_agent_attack>` to focus their attack on
a specific encoding.


Enhanced Flatten Wrapper
------------------------

Sampling from a flattened space can result in
`unexpected results <https://github.com/LLNL/Abmarl/issues/355>`_ when the sample
is later unflattened because the sampling distribution in the flattened space is
not the same as in the unflattened space. This is particularly important for Discrete
spaces, which are typically flattened as one-hot-encodings. To overcome these issues,
Abmarl's :ref:`FlattenWrapper <flatten_wrapper>` now flattens Discrete spaces as
an integer-Box of a single element with bounds up to ``space.n``.


Easier-to-Read Grid Loading
---------------------------

When loading the grid from a file, empty spaces were previously represented as
zeros. This made the file difficult for humans to read because the entities of
concern were hard to locate. Now, empty spaces can be zeros, dots, or underscores.
Here is a comparison:

.. code-block::
  
   # Zeros
   0 0 0 0 W 0 W W 0 W W 0 0 W W 0 W 0
   W 0 W 0 N 0 0 0 0 0 W 0 W W 0 0 0 0
   W W W W 0 W W 0 W 0 0 0 0 W W 0 W W
   0 W 0 0 0 W W 0 W 0 W W 0 0 0 0 0 0
   0 0 0 W 0 0 W W W 0 W 0 0 W 0 W W 0
   W W W W 0 W W W W W W W 0 W 0 T W 0
   0 0 0 0 0 W 0 0 0 0 0 0 0 W 0 W W 0
   0 W 0 W 0 W W W 0 W W 0 W W 0 W 0 0

   # Underscores
   _ _ _ _ W _ W W _ W W _ _ W W _ W _
   W _ W _ N _ _ _ _ _ W _ W W _ _ _ _
   W W W W _ W W _ W _ _ _ _ W W _ W W
   _ W _ _ _ W W _ W _ W W _ _ _ _ _ _
   _ _ _ W _ _ W W W _ W _ _ W _ W W _
   W W W W _ W W W W W W W _ W _ T W _
   _ _ _ _ _ W _ _ _ _ _ _ _ W _ W W _
   _ W _ W _ W W W _ W W _ W W _ W _ _

   # Dots
   . . . . W . W W . W W . . W W . W .
   W . W . N . . . . . W . W W . . . .
   W W W W . W W . W . . . . W W . W W
   . W . . . W W . W . W W . . . . . .
   . . . W . . W W W . W . . W . W W .
   W W W W . W W W W W W W . W . T W .
   . . . . . W . . . . . . . W . W W .
   . W . W . W W W . W W . W W . W . .

  


Miscellaneous
-------------

* New :ref:`AbsolutePositionObserver <gridworld_absolute_position_observer>` reports
  the agent's absolute position in the grid. This can be used in conjuction with the
  already-supported :ref:`Observers <gridworld_single_observer>` because the key is
  "position".
* The ``local_dir`` parameter in the configuration files will create a directory for
  the output files so that they are located under ``<local_dir>/abmarl_results/``.
  This behavior is consistente between the trainer and debugger. If no parameter
  is specified, Abmarl uses the home directory.
* A :ref:`PrincipleAgent's <api_principle_agent>` ``active`` property can be now
  directly set, giving components better control over the "done-state" of an agent.
* :ref:`Component Wrappers <gridworld_wrappers>` now wrap the null observation
  and null action of the agents in their underlying components.
* The :ref:`GymWrapper <gym_external>` now works with simulations with multiple entities
  as long as there is only a single :ref:`Learning Agent <api_agent>`.
* Abmarl supports ray version 2.0.
