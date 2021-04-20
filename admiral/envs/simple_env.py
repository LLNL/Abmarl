
import numpy as np

from admiral.envs.components.state import GridPositionState, LifeState
from admiral.envs.components.actor import AttackActor

from admiral.envs import AgentBasedSimulation, Agent

class MySimpleEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        
        # state
        self.position_state = GridPositionState(**kwargs)
        self.life_state = LifeState(**kwargs)

        # actor
        self.attack_actor = AttackActor(**kwargs)

        # observers


    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if action == 0: # don't move anywhere
                pass
            if action == 1: # move up
                self.position_state.modify_position(agent, np.array([0, 1]))
            if action == 2: # move up-right
                self.position_state.modify_position(agent, np.array([1, 1]))
            if action == 3: # move right
                self.position_state.modify_position(agent, np.array([1, 0]))
            if action == 4: # move down-right
                self.position_state.modify_position(agent, np.array([1, -1]))
            if action == 5: # move down
                self.position_state.modify_position(agent, np.array([0, -1]))
            if action == 6: # move down-left
                self.position_state.modify_position(agent, np.array([-1, -1]))
            if action == 7: # move left
                self.position_state.modify_position(agent, np.array([-1, 0]))
            if action == 8: # move up-left
                self.position_state.modify_position(agent, np.array([-1, 1]))
            if action == 9: # attack
                attacked_agent = self.attack_actor.process_attack(agent, 1)