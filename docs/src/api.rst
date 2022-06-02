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

.. _api_dynamic_sim:

.. autoclass:: abmarl.sim.DynamicOrderSimulation
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

.. _api_dynamic_man:

.. autoclass:: abmarl.managers.DynamicOrderManager
	:members:
	:undoc-members:


.. _api_gym_wrapper:

Abmarl External Integration
---------------------------

.. autoclass:: abmarl.external.GymWrapper
	:members:

.. _api_ma_wrapper:

.. autoclass:: abmarl.external.MultiAgentWrapper
	:members:



Abmarl GridWorld Simulation Framework
-------------------------------------

Base
````

.. _api_gridworld_sim:

.. autoclass:: abmarl.sim.gridworld.base.GridWorldSimulation
	:members:
	:undoc-members:

.. _api_gridworld_base:

.. autoclass:: abmarl.sim.gridworld.base.GridWorldBaseComponent
	:members:
	:undoc-members:

.. _api_gridworld_grid:

.. autoclass:: abmarl.sim.gridworld.grid.Grid
	:members:
	:undoc-members:


Agents
``````

.. _api_gridworld_agent:

.. autoclass:: abmarl.sim.gridworld.agent.GridWorldAgent
	:members:
	:undoc-members:

.. _api_gridworld_agent_observing:

.. autoclass:: abmarl.sim.gridworld.agent.GridObservingAgent
	:members:
	:undoc-members:

.. _api_gridworld_agent_moving:

.. autoclass:: abmarl.sim.gridworld.agent.MovingAgent
	:members:
	:undoc-members:

.. _api_gridworld_agent_health:

.. autoclass:: abmarl.sim.gridworld.agent.HealthAgent
	:members:
	:undoc-members:

.. _api_gridworld_agent_attack:

.. autoclass:: abmarl.sim.gridworld.agent.AttackingAgent
	:members:
	:undoc-members:


State
`````

.. _api_gridworld_statebase:

.. autoclass:: abmarl.sim.gridworld.state.StateBaseComponent
	:members:
	:undoc-members:

.. _api_gridworld_state_position:

.. autoclass:: abmarl.sim.gridworld.state.PositionState
	:members:
	:undoc-members:

.. _api_gridworld_state_health:

.. autoclass:: abmarl.sim.gridworld.state.HealthState
	:members:
	:undoc-members:


Actors
``````

.. _api_gridworld_actor:

.. autoclass:: abmarl.sim.gridworld.actor.ActorBaseComponent
	:members:
	:undoc-members:

.. _api_gridworld_actor_move:

.. autoclass:: abmarl.sim.gridworld.actor.MoveActor
	:members:
	:undoc-members:

.. _api_gridworld_actor_attack:

.. autoclass:: abmarl.sim.gridworld.actor.AttackActor
	:members:
	:undoc-members:


Observers
`````````

.. _api_gridworld_observer:

.. autoclass:: abmarl.sim.gridworld.observer.ObserverBaseComponent
	:members:
	:undoc-members:

.. _api_gridworld_observer_single:

.. autoclass:: abmarl.sim.gridworld.observer.SingleGridObserver
	:members:
	:undoc-members:

.. _api_gridworld_observer_multi:

.. autoclass:: abmarl.sim.gridworld.observer.MultiGridObserver
	:members:
	:undoc-members:


Done
````

.. _api_gridworld_done:

.. autoclass:: abmarl.sim.gridworld.done.DoneBaseComponent
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.done.ActiveDone
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.done.OneTeamRemainingDone
	:members:
	:undoc-members:


Wrappers
````````

.. _api_gridworld_wrappers:

.. autoclass:: abmarl.sim.gridworld.wrapper.ComponentWrapper
	:members:
	:undoc-members:

.. _api_gridworld_actor_wrappers:

.. autoclass:: abmarl.sim.gridworld.wrapper.ActorWrapper
	:members:
	:undoc-members:

.. _api_gridworld_ravel_action_wrappers:

.. autoclass:: abmarl.sim.gridworld.wrapper.RavelActionWrapper
	:members:
	:undoc-members:
