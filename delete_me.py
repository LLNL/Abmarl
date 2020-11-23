
# # Simplified implementation
# # %% Resource class manages PART of the state of the environment:

# class ResourceModule:
#     def __init__(self):
#         self.internal_state = "internal state"
    
#     def reset(self):
#         self.internal_state = "reset internal state"

# class MyEnv:
#     def __init__(self):
#         self.resources = ResourceModule()
    
#     def reset(self):
#         self.resources.reset()

# env = MyEnv()
# assert env.resources.internal_state == "internal state"
# env.reset()
# assert env.resources.internal_state == "reset internal state"

# # %% Some agents can harvest resources

# class ResourceModule:
#     def __init__(self):
#         self.internal_state = "internal state"
    
#     def reset(self):
#         self.internal_state = "reset internal state"

#     def harvest(self):
#         self.internal_state = "harvested internal state"
    
#     def regrow(self):
#         self.internal_state = "regrown internal state"

# class MyEnv:
#     def __init__(self):
#         self.resources = ResourceModule()
    
#     def reset(self):
#         self.resources.reset()
    
#     def step(self):
#         self.resources.harvest()
#         state1 = self.resources.internal_state
#         self.resources.regrow()
#         state2 = self.resources.internal_state
#         return state1 + " " + state2 

# env = MyEnv()
# assert env.resources.internal_state == "internal state"
# env.reset()
# assert env.resources.internal_state == "reset internal state"
# assert env.step() == "harvested internal state regrown internal state"

# env = MyEnv()

# # %% Some agents can observe a (sub)state of the environment

# class ResourceModule:
#     def __init__(self):
#         self.internal_state = "internal state"
    
#     def reset(self):
#         self.internal_state = "reset internal state"

#     def harvest(self):
#         self.internal_state = "harvested internal state"
    
#     def regrow(self):
#         self.internal_state = "regrown internal state"
    
#     def get_obs(self, agent_id):
#         return self.internal_state + " to " + agent_id

# class MyEnv:
#     def __init__(self):
#         self.resources = ResourceModule()
    
#     def reset(self):
#         self.resources.reset()
    
#     def step(self):
#         self.resources.harvest()
#         state1 = self.resources.internal_state
#         self.resources.regrow()
#         state2 = self.resources.internal_state
#         return state1 + " " + state2 
    
#     def get_obs(self, agent_id):
#         return self.resources.get_obs(agent_id)

# env = MyEnv()
# assert env.resources.internal_state == "internal state"
# env.reset()
# assert env.resources.internal_state == "reset internal state"
# assert env.step() == "harvested internal state regrown internal state"
# assert env.get_obs("agent1") == "regrown internal state to agent1"

# # %% Now we can incorporate some finer agent support

# class ResourceAgent:
#     pass

# class ResourceHarvestingAgent(ResourceAgent):
#     def __init__(self, harvest_amount=0.24):
#         self.harvest_amount = harvest_amount

# class ResourceObservingAgent(ResourceAgent):
#     def __init__(self, view=3):
#         self.view = view

# class ResourceHarvestingAndObservingAgent(ResourceHarvestingAgent, ResourceObservingAgent):
#     pass

# class ResourceModule:
#     def __init__(self, agents):
#         self.internal_state = "internal state"
#         self.agents = agents
    
#     def reset(self):
#         return "Reset the state"

#     def harvest(self):
#         return {agent_id: agent_id + " harvest" for agent_id in self.agents}
    
#     def regrow(self):
#         return "Regrow the resources"
    
#     def get_obs(self, agent_id):
#         return "observation for " + agent_id

# class MyEnv:
#     def __init__(self, **kwargs):
#         self.agents = kwargs['agents']
#         self.resources = ResourceModule(**kwargs)
    
#     def reset(self):
#         return self.resources.reset()
    
#     def step(self):
#         state1 = self.resources.harvest()
#         state2 = self.resources.regrow()
#         return state1, state2
    
#     def get_obs(self, agent_id):
#         return self.resources.get_obs(agent_id)

# agents = {
#     'agent0': ResourceHarvestingAgent(), # only harvest
#     'agent1': ResourceHarvestingAgent(), # only harvest
#     'agent2': ResourceObservingAgent(), # only observe
#     'agent3': ResourceHarvestingAndObservingAgent(), # harvest and observe
# }

# env = MyEnv(agents=agents)
# assert env.resources.internal_state == "internal state"
# assert env.reset() == "Reset the state"
# assert env.step()[0] == {
#     'agent0': 'agent0 harvest',
#     'agent1': 'agent1 harvest',
#     'agent2': 'agent2 harvest',
#     'agent3': 'agent3 harvest',
# }
# assert env.step()[1] == "Regrow the resources"
# assert env.get_obs("agent0") == "observation for agent0"
# assert env.get_obs("agent1") == "observation for agent1"
# assert env.get_obs("agent2") == "observation for agent2"
# assert env.get_obs("agent3") == "observation for agent3"
# assert env.agents == env.resources.agents


# # %% Impose the restrictions

# class ResourceAgent:
#     pass

# class ResourceHarvestingAgent(ResourceAgent):
#     def __init__(self, harvest_amount=0.24):
#         self.harvest_amount = harvest_amount

# class ResourceObservingAgent(ResourceAgent):
#     def __init__(self, view=3):
#         self.view = view

# class ResourceHarvestingAndObservingAgent(ResourceHarvestingAgent, ResourceObservingAgent):
#     pass

# class ResourceModule:
#     def __init__(self, agents):
#         self.internal_state = "internal state"
#         self.agents = agents
    
#     def reset(self):
#         return "Reset the state"

#     def harvest(self):
#         return {agent_id: agent_id + " harvest" for agent_id, agent in self.agents.items() if isinstance(agent, ResourceHarvestingAgent)}
    
#     def regrow(self):
#         return "Regrow the resources"
    
#     def get_obs(self, agent_id):
#         if isinstance(self.agents[agent_id], ResourceObservingAgent):
#             return "observation for " + agent_id

# class MyEnv:
#     def __init__(self, **kwargs):
#         self.agents = kwargs['agents']
#         self.resources = ResourceModule(**kwargs)
    
#     def reset(self):
#         return self.resources.reset()
    
#     def step(self):
#         state1 = self.resources.harvest()
#         state2 = self.resources.regrow()
#         return state1, state2
    
#     def get_obs(self, agent_id):
#         return self.resources.get_obs(agent_id)

# agents = {
#     'agent0': ResourceHarvestingAgent(), # only harvest
#     'agent1': ResourceHarvestingAgent(), # only harvest
#     'agent2': ResourceObservingAgent(), # only observe
#     'agent3': ResourceHarvestingAndObservingAgent(), # harvest and observe
# }

# env = MyEnv(agents=agents)
# assert env.resources.internal_state == "internal state"
# assert env.reset() == "Reset the state"
# assert env.step()[0] == {
#     'agent0': 'agent0 harvest',
#     'agent1': 'agent1 harvest',
#     'agent3': 'agent3 harvest',
# }
# assert env.step()[1] == "Regrow the resources"
# assert env.get_obs("agent2") == "observation for agent2"
# assert env.get_obs("agent3") == "observation for agent3"
# assert env.agents == env.resources.agents

# %% Apply the restrictions to the observation and action spaces of the agents

class Agent:
    def __init__(self, observation_space=None, action_space=None):
        self.observation_space = {} if observation_space is None else observation_space
        self.action_space = {} if action_space is None else action_space

    @property
    def configured(self):
        return self.observation_space and self.action_space

class ResourceAgent(Agent):
    pass

class ResourceHarvestingAgent(ResourceAgent):
    def __init__(self, harvest_amount=0.24):
        super().__init__()
        self.harvest_amount = harvest_amount

class ResourceObservingAgent(ResourceAgent):
    def __init__(self, view=3):
        super().__init__()
        self.view = view

class ResourceHarvestingAndObservingAgent(ResourceHarvestingAgent, ResourceObservingAgent):
    pass

class ResourceModule:
    def __init__(self, agents):
        self.internal_state = "internal state"
        self.agents = agents

        for agent in self.agents.values():
            if isinstance(agent, ResourceHarvestingAgent):
                agent.action_space['harvest'] = "Action space acquired"
            if isinstance(agent, ResourceObservingAgent):
                agent.observation_space['resources'] = "Observation space acquired"
    
    def reset(self):
        return "Reset the state"

    def harvest(self):
        return {agent_id: agent_id + " harvest" for agent_id, agent in self.agents.items() if isinstance(agent, ResourceHarvestingAgent)}
    
    def regrow(self):
        return "Regrow the resources"
    
    def get_obs(self, agent_id):
        if isinstance(self.agents[agent_id], ResourceObservingAgent):
            return "observation for " + agent_id

class MyEnvInterface:
    def __init__(self, **kwargs):
        # TODO: Add this configured check to the environment interface.
        self.agents = kwargs['agents']
        for agent in self.agents.values():
            assert agent.configured

class MyEnv(MyEnvInterface):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.resources = ResourceModule(**kwargs)

        # Cheat to fill spaces
        for agent in self.agents.values():
            if agent.observation_space == {}:
                agent.observation_space = "No Observation Space"
            if agent.action_space == {}:
                agent.action_space = "No Action Space"
        
        super().__init__(**kwargs)
    
    def reset(self):
        return self.resources.reset()
    
    def step(self):
        state1 = self.resources.harvest()
        state2 = self.resources.regrow()
        return state1, state2
    
    def get_obs(self, agent_id):
        return self.resources.get_obs(agent_id)

agents = {
    'agent0': ResourceHarvestingAgent(), # only harvest
    'agent1': ResourceHarvestingAgent(), # only harvest
    'agent2': ResourceObservingAgent(), # only observe
    'agent3': ResourceHarvestingAndObservingAgent(), # harvest and observe
    'agent4': Agent(), # Some other kind of agent
}

# Do not EXPECT agents to be configured
env = MyEnv(agents=agents)

assert agents['agent0'].configured
assert agents['agent0'].action_space['harvest'] == "Action space acquired"
assert agents['agent0'].observation_space == "No Observation Space"
assert agents['agent1'].configured
assert agents['agent1'].action_space['harvest']  == "Action space acquired"
assert agents['agent1'].observation_space == "No Observation Space"
assert agents['agent2'].configured
assert agents['agent2'].action_space == "No Action Space"
assert agents['agent2'].observation_space['resources'] == "Observation space acquired"
assert agents['agent3'].configured
assert agents['agent3'].action_space['harvest']  == "Action space acquired"
assert agents['agent3'].observation_space['resources'] == "Observation space acquired"
assert agents['agent4'].configured
assert agents['agent4'].action_space == "No Action Space"
assert agents['agent4'].observation_space == "No Observation Space"

assert env.resources.internal_state == "internal state"
assert env.reset() == "Reset the state"
assert env.step()[0] == {
    'agent0': 'agent0 harvest',
    'agent1': 'agent1 harvest',
    'agent3': 'agent3 harvest',
}
assert env.step()[1] == "Regrow the resources"
assert env.get_obs("agent0") is None
assert env.get_obs("agent1") is None
assert env.get_obs("agent2") == "observation for agent2"
assert env.get_obs("agent3") == "observation for agent3"
assert env.get_obs("agent4") is None
assert env.agents == env.resources.agents
assert agents == env.resources.agents

# %%
