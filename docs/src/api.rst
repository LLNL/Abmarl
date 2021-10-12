.. Abmarl documentation API.

Abmarl API Specification
=========================


Abmarl Simulations
------------------

.. _api_agent:

.. autoclass:: abmarl.sim.PrincipleAgent
	:members:
	:undoc-members:

.. _api_observing_agent:

.. autoclass:: abmarl.sim.ObservingAgent
	:members:
	:undoc-members:

.. _api_acting_agent:

.. autoclass:: abmarl.sim.ActingAgent
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.Agent
	:members:
	:undoc-members:
	:show-inheritance:

.. _api_abs:

.. autoclass:: abmarl.sim.AgentBasedSimulation
	:members:
	:undoc-members:


.. _api_sim:

Abmarl Simulation Managers
--------------------------

.. autoclass:: abmarl.managers.SimulationManager
	:members:
	:undoc-members:

.. _api_turn_based:

.. autoclass:: abmarl.managers.TurnBasedManager
	:members:
	:undoc-members:

.. _api_all_step:

.. autoclass:: abmarl.managers.AllStepManager
	:members:
	:undoc-members:


.. _api_gym_wrapper:

Abmarl External Integration
---------------------------

.. autoclass:: abmarl.external.GymWrapper
	:members:
	:undoc-members:

.. _api_ma_wrapper:

.. autoclass:: abmarl.external.MultiAgentWrapper
	:members:
	:undoc-members:



Abmarl GridWorld Simulation Framework
-------------------------------------

Base
````

.. autoclass:: abmarl.sim.gridworld.base.GridWorldSimulation
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.base.GridWorldBaseComponent
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.grid.Grid
	:members:
	:undoc-members:


Agent
`````

.. autoclass:: abmarl.sim.gridworld.agent.GridWorldAgent
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.agent.GridObservingAgent
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.agent.MovingAgent
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.agent.HealthAgent
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.agent.AttackingAgent
	:members:
	:undoc-members:


State
`````

.. autoclass:: abmarl.sim.gridworld.state.StateBaseComponent
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.state.PositionState
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.state.HealthState
	:members:
	:undoc-members:


Actor
`````

.. autoclass:: abmarl.sim.gridworld.actor.ActorBaseComponent
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.actor.MoveActor
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.actor.AttackActor
	:members:
	:undoc-members:


Observer
````````

.. autoclass:: abmarl.sim.gridworld.observer.ObserverBaseComponent
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.observer.SingleGridObserver
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.observer.MultiGridObserver
	:members:
	:undoc-members:


Done
````

.. autoclass:: abmarl.sim.gridworld.done.DoneBaseComponent
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.done.ActiveDone
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.done.OneTeamRemainingDone
	:members:
	:undoc-members:
