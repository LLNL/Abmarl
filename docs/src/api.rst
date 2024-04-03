.. Abmarl documentation API.

Abmarl API Specification
=========================


Abmarl Simulations
------------------

.. _api_principle_agent:

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

.. _api_agent:

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


.. _api_wrappers:

Abmarl Wrappers
---------------

.. autoclass:: abmarl.sim.wrappers.Wrapper
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.wrappers.SARWrapper
	:members:
	:undoc-members:

.. _api_ravel_wrapper:

.. autoclass:: abmarl.sim.wrappers.RavelDiscreteWrapper
	:members:
	:undoc-members:

.. _api_flatten_wrapper:

.. autoclass:: abmarl.sim.wrappers.FlattenWrapper
	:members:
	:undoc-members:

.. _api_super_agent_wrapper:

.. autoclass:: abmarl.sim.wrappers.SuperAgentWrapper
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

.. _api_openspiel_wrapper:

.. autoclass:: abmarl.external.OpenSpielWrapper
	:members:



Abmarl GridWorld Simulation Framework
-------------------------------------

Base
````

.. _api_gridworld_sim:

.. autoclass:: abmarl.sim.gridworld.base.GridWorldSimulation
	:members:
	:undoc-members:

.. _api_gridworld_smart_sim:

.. autoclass:: abmarl.sim.gridworld.smart.SmartGridWorldSimulation
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

.. _api_gridworld_register:

.. autofunction:: abmarl.sim.gridworld.registry.register


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

.. _api_gridworld_agent_orientation:

.. autoclass:: abmarl.sim.gridworld.agent.OrientationAgent
	:members:
	:undoc-members:

.. _api_gridworld_agent_attack:

.. autoclass:: abmarl.sim.gridworld.agent.AttackingAgent
	:members:
	:undoc-members:

.. _api_gridworld_agent_ammo:

.. autoclass:: abmarl.sim.gridworld.agent.AmmoAgent
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

.. _api_gridworld_state_position_maze:

.. autoclass:: abmarl.sim.gridworld.state.MazePlacementState
	:members:
	:undoc-members:

.. _api_gridworld_state_position_target_barriers_free:

.. autoclass:: abmarl.sim.gridworld.state.TargetBarriersFreePlacementState
	:members:
	:undoc-members:

.. _api_gridworld_state_health:

.. autoclass:: abmarl.sim.gridworld.state.HealthState
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.state.AmmoState
	:members:
	:undoc-members:

.. autoclass:: abmarl.sim.gridworld.state.OrientationState
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

.. _api_gridworld_actor_cross_move:

.. autoclass:: abmarl.sim.gridworld.actor.CrossMoveActor
	:members:
	:undoc-members:

.. _api_gridworld_actor_drift_move:

.. autoclass:: abmarl.sim.gridworld.actor.DriftMoveActor
	:members:
	:undoc-members:

.. _api_gridworld_actor_attack:

.. autoclass:: abmarl.sim.gridworld.actor.AttackActorBaseComponent
	:members:
	:undoc-members:

.. _api_gridworld_actor_binary_attack:

.. autoclass:: abmarl.sim.gridworld.actor.BinaryAttackActor
	:members:
	:undoc-members:

.. _api_gridworld_actor_encoding_attack:

.. autoclass:: abmarl.sim.gridworld.actor.EncodingBasedAttackActor
	:members:
	:undoc-members:

.. _api_gridworld_actor_selective_attack:

.. autoclass:: abmarl.sim.gridworld.actor.SelectiveAttackActor
	:members:
	:undoc-members:

.. _api_gridworld_actor_restricted_selective_attack:

.. autoclass:: abmarl.sim.gridworld.actor.RestrictedSelectiveAttackActor
	:members:
	:undoc-members:


Observers
`````````

.. _api_gridworld_observer:

.. autoclass:: abmarl.sim.gridworld.observer.ObserverBaseComponent
	:members:
	:undoc-members:

.. _api_gridworld_observer_ammo:

.. autoclass:: abmarl.sim.gridworld.observer.AmmoObserver
	:members:
	:undoc-members:

.. _api_gridworld_observer_absolute_position:

.. autoclass:: abmarl.sim.gridworld.observer.AbsolutePositionObserver
	:members:
	:undoc-members:

.. _api_gridworld_observer_absolute_encoding:

.. autoclass:: abmarl.sim.gridworld.observer.AbsoluteEncodingObserver
	:members:
	:undoc-members:

.. _api_gridworld_observer_position_centered:

.. autoclass:: abmarl.sim.gridworld.observer.PositionCenteredEncodingObserver
	:members:
	:undoc-members:

.. _api_gridworld_observer_position_centered_stacked:

.. autoclass:: abmarl.sim.gridworld.observer.StackedPositionCenteredEncodingObserver
	:members:
	:undoc-members:


Done
````

.. _api_gridworld_done:

.. autoclass:: abmarl.sim.gridworld.done.DoneBaseComponent
	:members:
	:undoc-members:

.. _api_gridworld_done_active:

.. autoclass:: abmarl.sim.gridworld.done.ActiveDone
	:members:
	:undoc-members:

.. _api_gridworld_done_one_team_remaining:

.. autoclass:: abmarl.sim.gridworld.done.OneTeamRemainingDone
	:members:
	:undoc-members:

.. _api_gridworld_done_target_agent_overlap:

.. autoclass:: abmarl.sim.gridworld.done.TargetAgentOverlapDone
	:members:
	:undoc-members:

.. _api_gridworld_done_target_agent_inactive:

.. autoclass:: abmarl.sim.gridworld.done.TargetAgentInactiveDone
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

.. _api_gridworld_exclusive_channel_action_wrappers:

.. autoclass:: abmarl.sim.gridworld.wrapper.ExclusiveChannelActionWrapper
	:members:
	:undoc-members:


.. _api_multi_policy_trainer:

Abmarl Trainers
---------------

.. autoclass:: abmarl.trainers.MultiPolicyTrainer
	:members:
	:undoc-members:

.. _api_single_policy_trainer:

.. autoclass:: abmarl.trainers.SinglePolicyTrainer
	:members:
	:undoc-members:

.. _api_monte_carlo_trainer:

.. autoclass:: abmarl.trainers.monte_carlo.OnPolicyMonteCarloTrainer
	:members:
	:undoc-members:

.. _api_debug_trainer:

.. autoclass:: abmarl.trainers.DebugTrainer
	:members:
	:undoc-members:

