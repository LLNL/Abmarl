
from admiral.component_envs.death_life import HealthAgent, LifeAgent
from admiral.component_envs.death_life import DyingComponent

def test_health_agents():
    agents = {
        'agent0': HealthAgent(id='agent0', min_health=0.0, max_health=5.0, initial_health=3.4),
        'agent1': HealthAgent(id='agent1', min_health=0.0, max_health=5.0, initial_health=2.4),
        'agent2': HealthAgent(id='agent2', min_health=0.0, max_health=5.0),
        'agent3': HealthAgent(id='agent3', min_health=0.0, max_health=5.0),
    }

    assert agents['agent0'].min_health == 0.0
    assert agents['agent0'].max_health == 5.0
    assert agents['agent0'].initial_health == 3.4
    assert agents['agent1'].min_health == 0.0
    assert agents['agent1'].max_health == 5.0
    assert agents['agent1'].initial_health == 2.4
    assert agents['agent2'].min_health == 0.0
    assert agents['agent2'].max_health == 5.0
    assert agents['agent3'].min_health == 0.0
    assert agents['agent3'].max_health == 5.0

def test_dying_component():
    agents = {
        'agent0': LifeAgent(id='agent0', min_health=0.0, max_health=5.0, initial_health=3.4),
        'agent1': LifeAgent(id='agent1', min_health=0.0, max_health=5.0, initial_health=2.4),
        'agent2': LifeAgent(id='agent2', min_health=0.0, max_health=5.0),
        'agent3': LifeAgent(id='agent3', min_health=0.0, max_health=5.0),
    }

    assert agents['agent0'].min_health == 0.0
    assert agents['agent0'].max_health == 5.0
    assert agents['agent0'].initial_health == 3.4
    assert agents['agent0'].is_alive == True
    assert agents['agent1'].min_health == 0.0
    assert agents['agent1'].max_health == 5.0
    assert agents['agent1'].initial_health == 2.4
    assert agents['agent1'].is_alive == True
    assert agents['agent2'].min_health == 0.0
    assert agents['agent2'].max_health == 5.0
    assert agents['agent2'].is_alive == True
    assert agents['agent3'].min_health == 0.0
    assert agents['agent3'].max_health == 5.0
    assert agents['agent3'].is_alive == True

    component = DyingComponent(agents=agents, entropy=0.5)
    component.reset()
    assert agents['agent0'].health == 3.4
    assert agents['agent1'].health == 2.4
    assert 0.0 <= agents['agent2'].health <= 5.0
    assert 0.0 <= agents['agent3'].health <= 5.0

    component.apply_entropy(agents['agent0'])
    component.apply_entropy(agents['agent1'])
    assert agents['agent0'].health == 2.9
    assert agents['agent1'].health == 1.9

    for _ in range(10):
        component.apply_entropy(agents['agent0'])
        component.apply_entropy(agents['agent1'])
        component.apply_entropy(agents['agent2'])
        component.apply_entropy(agents['agent3'])

    assert agents['agent0'].is_alive is False
    assert agents['agent1'].is_alive is False
    assert agents['agent2'].is_alive is False
    assert agents['agent3'].is_alive is False

    component.reset()
    assert agents['agent0'].min_health == 0.0
    assert agents['agent0'].max_health == 5.0
    assert agents['agent0'].initial_health == 3.4
    assert agents['agent0'].is_alive == True
    assert agents['agent1'].min_health == 0.0
    assert agents['agent1'].max_health == 5.0
    assert agents['agent1'].initial_health == 2.4
    assert agents['agent1'].is_alive == True
    assert agents['agent2'].min_health == 0.0
    assert agents['agent2'].max_health == 5.0
    assert agents['agent2'].is_alive == True
    assert agents['agent3'].min_health == 0.0
    assert agents['agent3'].max_health == 5.0
    assert agents['agent3'].is_alive == True
