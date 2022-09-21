
import numpy as np
import pytest

from abmarl.sim.gridworld.agent import GridWorldAgent, GridObservingAgent, MovingAgent, \
    HealthAgent, AttackingAgent
from abmarl.sim import PrincipleAgent, ActingAgent, ObservingAgent


def test_grid_world_agent():
    agent = GridWorldAgent(
        id='agent',
        initial_position=np.array([2, 2]),
        blocking=True,
        encoding=4
    )
    assert isinstance(agent, PrincipleAgent)
    assert agent.id == 'agent'
    np.testing.assert_array_equal(agent.initial_position, np.array([2, 2]))
    assert agent.blocking
    assert agent.encoding == 4
    assert agent.render_shape == 'o'
    assert agent.render_color == 'gray'
    assert agent.configured

    # Encoding
    with pytest.raises(AssertionError):
        agent = GridWorldAgent(
            id='agent',
            encoding='2'
        )
    with pytest.raises(AssertionError):
        agent = GridWorldAgent(
            id='agent',
            encoding=-2
        )
    with pytest.raises(AssertionError):
        agent = GridWorldAgent(
            id='agent',
            encoding=-1
        )
    with pytest.raises(AssertionError):
        agent = GridWorldAgent(
            id='agent',
            encoding=0
        )

    # Initial position
    with pytest.raises(AssertionError):
        agent = GridWorldAgent(
            id='agent',
            initial_position=np.array(['0', '1'])
        )
    with pytest.raises(AssertionError):
        agent = GridWorldAgent(
            id='agent',
            initial_position=np.array([0, 1, 2])
        )
    with pytest.raises(AssertionError):
        agent = GridWorldAgent(
            id='agent',
            initial_position=[0, 1, 2]
        )

    # Blocking
    with pytest.raises(AssertionError):
        agent = GridWorldAgent(
            id='agent',
            blocking=1
        )

    # Render shape
    with pytest.raises(AssertionError):
        agent = GridWorldAgent(
            id='agent',
            render_shape='circle'
        )


def test_grid_observing_agent():
    agent = GridObservingAgent(
        id='agent',
        encoding=1,
        view_range=3
    )
    assert isinstance(agent, ObservingAgent)
    assert isinstance(agent, GridWorldAgent)
    assert agent.view_range == 3

    # View range
    with pytest.raises(AssertionError):
        agent = GridObservingAgent(
            id='agent',
            encoding=1,
            view_range=-1
        )


def test_moving_agent():
    agent = MovingAgent(
        id='agent',
        encoding=1,
        move_range=2
    )
    assert isinstance(agent, ActingAgent)
    assert isinstance(agent, GridWorldAgent)
    assert agent.move_range == 2

    with pytest.raises(AssertionError):
        agent = MovingAgent(
            id='agent',
            encoding=1,
            move_range='1'
        )


def test_health_agent():
    agent = HealthAgent(
        id='agent',
        encoding=1,
        initial_health=0.45
    )
    assert isinstance(agent, GridWorldAgent)
    assert agent.initial_health == 0.45
    assert agent.configured

    with pytest.raises(AssertionError):
        agent = HealthAgent(
            id='agent',
            encoding=1,
            initial_health=2
        )


def test_attacking_agent():
    agent = AttackingAgent(
        id='agent',
        encoding=1,
        attack_range=3,
        attack_strength=0.6,
        attack_accuracy=0.95
    )
    assert isinstance(agent, ActingAgent)
    assert isinstance(agent, GridWorldAgent)
    assert agent.attack_range == 3
    assert agent.attack_strength == 0.6
    assert agent.attack_accuracy == 0.95

    with pytest.raises(AssertionError):
        agent = AttackingAgent(
            id='agent',
            encoding=1,
            attack_range=3.0,
            attack_strength=0.6,
            attack_accuracy=0.95
        )

    with pytest.raises(AssertionError):
        agent = AttackingAgent(
            id='agent',
            encoding=1,
            attack_range=3,
            attack_strength=2,
            attack_accuracy=0.95
        )

    with pytest.raises(AssertionError):
        agent = AttackingAgent(
            id='agent',
            encoding=1,
            attack_range=3,
            attack_strength=0.6,
            attack_accuracy=-0.3
        )

    with pytest.raises(AssertionError):
        agent = AttackingAgent(
            id='agent',
            encoding=1,
            attack_range=3,
            attack_strength=0.6,
            attack_accuracy=0.95,
            attack_count=-2
        )
