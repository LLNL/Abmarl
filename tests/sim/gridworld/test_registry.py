
import pytest

from abmarl.sim.gridworld.actor import MoveActor, CrossMoveActor, \
    BinaryAttackActor, EncodingBasedAttackActor, RestrictedSelectiveAttackActor, \
    SelectiveAttackActor
from abmarl.sim.gridworld.done import ActiveDone, TargetAgentDone, \
    OneTeamRemainingDone
from abmarl.sim.gridworld.observer import AbsoluteEncodingObserver, \
    PositionCenteredEncodingObserver, StackedPositionCenteredEncodingObserver, \
    AbsolutePositionObserver
from abmarl.sim.gridworld.state import PositionState, \
    TargetBarriersFreePlacementState, MazePlacementState, HealthState
from abmarl.sim.gridworld.registry import registry, register

from abmarl.examples.sim.comms_blocking import BroadcastingActor, BroadcastingState, \
    BroadcastObserver, AverageMessageDone, BroadcastingAgent


def test_built_in_registry():
    assert MoveActor in registry['actor'].values()
    assert CrossMoveActor in registry['actor'].values()
    assert BinaryAttackActor in registry['actor'].values()
    assert EncodingBasedAttackActor in registry['actor'].values()
    assert RestrictedSelectiveAttackActor in registry['actor'].values()
    assert SelectiveAttackActor in registry['actor'].values()

    assert ActiveDone in registry['done'].values()
    assert TargetAgentDone in registry['done'].values()
    assert OneTeamRemainingDone in registry['done'].values()

    assert AbsoluteEncodingObserver in registry['observer'].values()
    assert PositionCenteredEncodingObserver in registry['observer'].values()
    assert StackedPositionCenteredEncodingObserver in registry['observer'].values()
    assert AbsolutePositionObserver in registry['observer'].values()

    assert PositionState in registry['state'].values()
    assert TargetBarriersFreePlacementState in registry['state'].values()
    assert MazePlacementState in registry['state'].values()
    assert HealthState in registry['state'].values()


def test_custom_registrations():
    register(BroadcastingState)
    register(BroadcastingActor)
    register(BroadcastObserver)
    register(AverageMessageDone)

    assert BroadcastingState in registry['state'].values()
    assert BroadcastingActor in registry['actor'].values()
    assert BroadcastObserver in registry['observer'].values()
    assert AverageMessageDone in registry['done'].values()

    with pytest.raises(TypeError):
        register(BroadcastingAgent)
