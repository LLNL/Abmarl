
import numpy as np

from admiral.component_envs.attacking import GridAttackingEnv, AttackingTeamAgent

def test_attack_configurations():
    agents = {
        'agent0': AttackingTeamAgent(id='agent0', team=0, attack_range=1, starting_position=np.array([1, 1])),
        'agent1': AttackingTeamAgent(id='agent1', team=1, attack_range=1, starting_position=np.array([0, 1])),
        'agent2': AttackingTeamAgent(id='agent2', team=1, attack_range=1, starting_position=np.array([4, 2])),
        'agent3': AttackingTeamAgent(id='agent3', team=0, attack_range=1, starting_position=np.array([4, 3])),
        'agent4': AttackingTeamAgent(id='agent4', team=1, attack_range=0, starting_position=np.array([3, 2])),
        'agent5': AttackingTeamAgent(id='agent5', team=2, attack_range=2, starting_position=np.array([4, 0])),
    }
    env = GridAttackingEnv(
        region=5,
        agents=agents
    )
    for agent in env.agents.values():
        agent.position = agent.starting_position
    
    assert env.process_attack(env.agents['agent0']) == 'agent1'
    assert env.process_attack(env.agents['agent1']) == 'agent0'
    assert env.process_attack(env.agents['agent2']) == 'agent3'
    assert env.process_attack(env.agents['agent3']) == 'agent2'
    assert env.process_attack(env.agents['agent4']) == None
    assert env.process_attack(env.agents['agent5']) == 'agent2'
