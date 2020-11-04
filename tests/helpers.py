
from admiral.envs import AgentBasedSimulation
from admiral.envs import Agent

from gym.spaces import Discrete, MultiBinary, MultiDiscrete, Box, Dict, Tuple
import numpy as np

class FillInHelper(AgentBasedSimulation):
    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def get_obs(self, agent_id, **kwargs):
        pass

    def get_reward(self, agent_id, **kwargs):
        pass

    def get_done(self, agent_id, **kwargs):
        pass

    def get_all_done(self, **kwargs):
        pass

    def get_info(self, agent_id, **kwargs):
        pass

class MultiAgentEnv(FillInHelper):
    def __init__(self, num_agents=3):
        self.agents = {'agent' + str(i): Agent('agent'+str(i), 0, 0) for i in range(num_agents)}

    def reset(self):
        self.action = {agent.id: None for agent in self.agents.values()}

    def step(self, action_dict):
        for agent_id, action in action_dict.items():
            self.action[agent_id] = action
    
    def get_obs(self, agent_id, **kwargs):
        return "Obs from " + agent_id
    
    def get_reward(self, agent_id, **kwargs):
        return "Reward from " + agent_id
    
    def get_done(self, agent_id, **kwargs):
        return "Done from " + agent_id
    
    def get_all_done(self, **kwargs):
        return "Done from all agents and/or environment."
    
    def get_info(self, agent_id, **kwargs):
        return {'Action from ' + agent_id: self.action[agent_id]}

class MultiAgentGymSpacesEnv(MultiAgentEnv):
    def __init__(self):
        self.params = {'params': "there are none"}
        self.agents = {
            'agent0': Agent('agent0', MultiBinary(4), Tuple((Dict({'first': Discrete(4), 'second': Box(low=-1, high=3, shape=(2,), dtype=np.int)}), MultiBinary(3)))),
            'agent1': Agent('agent1', Box(low=0, high=1, shape=(1,), dtype=np.int), MultiDiscrete([4, 6, 2])),
            'agent2': Agent('agent2', MultiDiscrete([2, 2]), Dict({'alpha': MultiBinary(3)})),
            'agent3': Agent('agent3', Dict({'first': Discrete(4), 'second': Box(low=-1, high=3, shape=(2,), dtype=np.int)}), Tuple((Discrete(3), MultiDiscrete([10, 10]), Discrete(2))))
        }
    
    def get_obs(self, agent_id, **kwargs):
        if agent_id == 'agent0':
            return [0, 0, 0, 1]
        elif agent_id == 'agent1':
            return 0
        elif agent_id == 'agent2':
            return [1, 0]
        elif agent_id == 'agent3':
            return {'first': 1, 'second': [3, 1]}
    
    def get_info(self, agent_id, **kwargs):
        return self.action[agent_id]
