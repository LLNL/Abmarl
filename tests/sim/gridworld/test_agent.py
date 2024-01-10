
import numpy as np
import pytest

from abmarl.sim.gridworld.agent import GridWorldAgent, GridObservingAgent, MovingAgent, \
    HealthAgent, AttackingAgent, AmmoAgent, AmmoObservingAgent, OrientationAgent
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
    assert agent.render_size == 200
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

    # Render size
    with pytest.raises(AssertionError):
        agent = GridWorldAgent(
            id='agent',
            render_size=-3
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
            simultaneous_attacks=-2
        )


def test_ammo_agent():
    agent = AmmoAgent(
        id='agent',
        encoding=1,
        initial_ammo=4
    )
    assert isinstance(agent, GridWorldAgent)
    assert agent.initial_ammo == 4
    assert agent.configured

    agent = AmmoAgent(
        id='agent',
        encoding=1,
        initial_ammo=-2
    )
    assert agent.initial_ammo == -2

    with pytest.raises(AssertionError):
        agent = AmmoAgent(
            id='agent',
            encoding=1,
            initial_ammo=2.4
        )

    with pytest.raises(AssertionError):
        agent = AmmoAgent(
            id='agent',
            encoding=1,
        )


def test_ammo_observing_agent():
    class CustomAmmoObservingAgent(AmmoAgent, ObservingAgent): pass

    agent = AmmoObservingAgent(
        id='agent',
        encoding=1,
        initial_ammo=4
    )
    assert isinstance(agent, AmmoAgent)
    assert isinstance(agent, ObservingAgent)
    assert isinstance(agent, GridWorldAgent)

    agent = CustomAmmoObservingAgent(
        id='agent',
        encoding=1,
        initial_ammo=2
    )
    assert isinstance(agent, AmmoObservingAgent)

    with pytest.raises(AssertionError):
        agent = AmmoObservingAgent(
            id='agent',
            encoding=1,
            initial_ammo=2.4
        )


def test_orientation_agent():
    agent = OrientationAgent(
        id='agent',
        encoding=1,
        initial_orientation=3
    )
    assert isinstance(agent, OrientationAgent)
    assert isinstance(agent, GridWorldAgent)
    assert agent.initial_orientation == 3

    agent.orientation = 2
    assert agent.orientation == 2

    with pytest.raises(AssertionError):
        OrientationAgent(
            id='agent',
            encoding=1,
            initial_orientation='left'
        )
        OrientationAgent(
            id='agent',
            encoding=1,
            initial_orientation='N'
        )
        OrientationAgent(
            id='agent',
            encoding=1,
            initial_orientation=0
        )
        OrientationAgent(
            id='agent',
            encoding=1,
            orientation='left'
        )
        OrientationAgent(
            id='agent',
            encoding=1,
            orientation='N'
        )
        OrientationAgent(
            id='agent',
            encoding=1,
            orientation=0
        )
