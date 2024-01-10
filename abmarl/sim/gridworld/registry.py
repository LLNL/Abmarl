
from .actor import ActorBaseComponent, MoveActor, CrossMoveActor, BinaryAttackActor, \
    EncodingBasedAttackActor, RestrictedSelectiveAttackActor, SelectiveAttackActor, \
    DriftMoveActor
from .done import DoneBaseComponent, ActiveDone, TargetAgentDone, OneTeamRemainingDone, \
    TargetDestroyedDone
from .observer import ObserverBaseComponent, AbsoluteEncodingObserver, \
    PositionCenteredEncodingObserver, StackedPositionCenteredEncodingObserver, \
    AbsolutePositionObserver, AmmoObserver
from .state import StateBaseComponent, PositionState, TargetBarriersFreePlacementState, \
    MazePlacementState, HealthState, AmmoState, OrientationState


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
        DriftMoveActor,
        BinaryAttackActor,
        EncodingBasedAttackActor,
        RestrictedSelectiveAttackActor,
        SelectiveAttackActor
    }, 'done': {
        ActiveDone,
        TargetAgentDone,
        TargetDestroyedDone,
        OneTeamRemainingDone
    }, 'observer': {
        AbsoluteEncodingObserver,
        PositionCenteredEncodingObserver,
        StackedPositionCenteredEncodingObserver,
        AbsolutePositionObserver,
        AmmoObserver
    }, 'state': {
        PositionState,
        TargetBarriersFreePlacementState,
        MazePlacementState,
        HealthState,
        AmmoState,
        OrientationState
    }
}


registry = {
    component_type: {component.__name__: component for component in components}
    for component_type, components in _registered_components.items()
}


def register(component):
    """
    Register a component.

    Args:
        component: The component will be registered by its type (actor, done, observer,
            or state) and class name.
    """
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
