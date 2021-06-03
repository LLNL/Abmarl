from admiral.envs.components.agent import ComponentAgent as Agent
from admiral.envs.components.state import LifeState


def test_health_agents():
    agents = {
        'agent0': Agent(id='agent0', min_health=0.0, max_health=5.0, initial_health=3.4),
        'agent1': Agent(id='agent1', min_health=0.0, max_health=5.0, initial_health=2.4),
        'agent2': Agent(id='agent2', min_health=0.0, max_health=5.0),
        'agent3': Agent(id='agent3', min_health=0.0, max_health=5.0),
    }

    assert agents['agent0'].min_health == 0.0
    assert agents['agent0'].max_health == 5.0
    assert agents['agent0'].initial_health == 3.4
    assert agents['agent0'].is_alive
    assert agents['agent1'].min_health == 0.0
    assert agents['agent1'].max_health == 5.0
    assert agents['agent1'].initial_health == 2.4
    assert agents['agent1'].is_alive
    assert agents['agent2'].min_health == 0.0
    assert agents['agent2'].max_health == 5.0
    assert agents['agent2'].is_alive
    assert agents['agent3'].min_health == 0.0
    assert agents['agent3'].max_health == 5.0
    assert agents['agent3'].is_alive


def test_life_state():
    agents = {
        'agent0': Agent(id='agent0', min_health=0.0, max_health=5.0, initial_health=3.4),
        'agent1': Agent(id='agent1', min_health=0.0, max_health=5.0, initial_health=2.4),
        'agent2': Agent(id='agent2', min_health=0.0, max_health=5.0),
        'agent3': Agent(id='agent3', min_health=0.0, max_health=5.0),
    }

    assert agents['agent0'].min_health == 0.0
    assert agents['agent0'].max_health == 5.0
    assert agents['agent0'].initial_health == 3.4
    assert agents['agent0'].is_alive
    assert agents['agent1'].min_health == 0.0
    assert agents['agent1'].max_health == 5.0
    assert agents['agent1'].initial_health == 2.4
    assert agents['agent1'].is_alive
    assert agents['agent2'].min_health == 0.0
    assert agents['agent2'].max_health == 5.0
    assert agents['agent2'].is_alive
    assert agents['agent3'].min_health == 0.0
    assert agents['agent3'].max_health == 5.0
    assert agents['agent3'].is_alive

    state = LifeState(agents=agents, entropy=0.5)
    state.reset()

    assert agents['agent0'].health == 3.4
    assert agents['agent1'].health == 2.4
    assert 0.0 <= agents['agent2'].health <= 5.0
    assert 0.0 <= agents['agent3'].health <= 5.0

    state.apply_entropy(agents['agent0'])
    state.apply_entropy(agents['agent1'])
    assert agents['agent0'].health == 2.9
    assert agents['agent1'].health == 1.9

    for _ in range(10):
        state.apply_entropy(agents['agent0'])
        state.apply_entropy(agents['agent1'])
        state.apply_entropy(agents['agent2'])
        state.apply_entropy(agents['agent3'])

    assert agents['agent0'].is_alive is False
    assert agents['agent1'].is_alive is False
    assert agents['agent2'].is_alive is False
    assert agents['agent3'].is_alive is False

    state.reset()
    assert agents['agent0'].min_health == 0.0
    assert agents['agent0'].max_health == 5.0
    assert agents['agent0'].initial_health == 3.4
    assert agents['agent0'].is_alive
    assert agents['agent1'].min_health == 0.0
    assert agents['agent1'].max_health == 5.0
    assert agents['agent1'].initial_health == 2.4
    assert agents['agent1'].is_alive
    assert agents['agent2'].min_health == 0.0
    assert agents['agent2'].max_health == 5.0
    assert agents['agent2'].is_alive
    assert agents['agent3'].min_health == 0.0
    assert agents['agent3'].max_health == 5.0
    assert agents['agent3'].is_alive
