
from .actor import ActorBaseComponent, MoveActor, CrossMoveActor, BinaryAttackActor, \
    EncodingBasedAttackActor, RestrictedSelectiveAttackActor, SelectiveAttackActor
from .done import DoneBaseComponent, ActiveDone, TargetAgentDone, OneTeamRemainingDone
from .observer import ObserverBaseComponent, AbsoluteGridObserver, SingleGridObserver, \
    MultiGridObserver, AbsolutePositionObserver
from .state import StateBaseComponent, PositionState, TargetBarriersFreePlacementState, \
    MazePlacementState, HealthState


_subclass_check_mapping = {
    'actor': ActorBaseComponent,
    'done': DoneBaseComponent,
    'observer': ObserverBaseComponent,
    'state': StateBaseComponent
}

_registered_components = {
    'actor': {
        MoveActor,
        CrossMoveActor,
        BinaryAttackActor,
        EncodingBasedAttackActor,
        RestrictedSelectiveAttackActor,
        SelectiveAttackActor
    }, 'done': {
        ActiveDone,
        TargetAgentDone,
        OneTeamRemainingDone
    }, 'observer': {
        AbsoluteGridObserver,
        SingleGridObserver,
        MultiGridObserver,
        AbsolutePositionObserver
    }, 'state': {
        PositionState,
        TargetBarriersFreePlacementState,
        MazePlacementState,
        HealthState
    }
}


registry = {
    component_type: {component.__name__: component for component in components}
    for component_type, components in _registered_components.items()
}


def register(component):
    registered = False
    for component_type, base_component in _subclass_check_mapping.items():
        if issubclass(component, base_component):
            _registered_components[component_type].add(component)
            registry[component_type][component.__name__] = component
            registered = True
            break # Assumes that a component is a subclass of only one base component

    if not registered:
        raise TypeError(
            f"{component.__name__} must be an actor, done, state, or observer component."
        )
