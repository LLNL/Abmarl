
#%% Define skeleton of environment that implements an ABS interface
# import numpy as np

# from admiral.envs import AgentBasedSimulation

# # Let's define environment with a state that is made up of:
# # 1. Agent positions
# # 2. Agent health
# # 3. Agent team
# # 
# # Agents can move around in the environment
# # Agents can attack other agents from different team

# class Battle(AgentBasedSimulation):
#     def __init__(self, agents=None, region=None, **kwargs):
#         self.region = region
#         assert type(agents) is dict, "agents must be a dictionary"
#         self.agents = agents
    
#     def reset(self, **kwargs):
#         for agent in self.agents.values():
#             agent.position = np.random.randint(0, self.region, size=2)
    
#     def step(self, action_dict, **kwargs):
#         # Agents can move

#         # Agents can attack other agents from different team
#         pass

#     def render(self, **kwargs):
#         pass

#     def get_obs(self, agent_id, **kwargs):
#         pass

#     def get_reward(self, agent_id, **kwargs):
#         pass

#     def get_done(self, agent_id, **kwargs):
#         pass

#     def get_all_done(self, **kwargs):
#         pass

#     def get_info(self, agent_id, **kwargs):
#         pass

# env = Battle(agents={}, region=10)
# env.reset()

#%% Define component to handle the agents' positions and any modifications to the position.
# # Keep track of this in the agent object
# import numpy as np

# from admiral.envs import AgentBasedSimulation, Agent

# # Let's define environment with a state that is made up of:
# # 1. Agent positions
# # 2. Agent health
# # 3. Agent team
# # 
# # Agents can move around in the environment
# # Agents can attack other agents from different team

# from gym.spaces import MultiBinary, Box, Dict

# class GridPositionAgent(Agent):
#     def __init__(self, starting_position=None, **kwargs):
#         super().__init__(**kwargs)
#         self.starting_position = starting_position

# class MovingAndAttackingAgent(Agent):
#     def __init__(self, move_range=None, attack_range=None, **kwargs):
#         super().__init__(**kwargs)
#         self.move_range = move_range
#         self.attack_range = attack_range
        
#         self.action_space = Dict({
#             'move': Box(-self.move_range, self.move_range, (2,), np.int),
#             'attack': MultiBinary(1),
#         })

# class BattleAgent(GridPositionAgent, MovingAndAttackingAgent): pass

# class GridPositionComponent:
#     def __init__(self, agents=None, region=None, **kwargs):
#         for agent in agents.values():
#             assert isinstance(agent, GridPositionAgent)
#         self.agents = agents
#         self.region = region
    
#     def reset(self, agent_id, **kwargs):
#         agent = self.agents[agent_id]
#         if agent.starting_position is not None:
#             agent.position = agent.starting_position
#         else:
#             agent.position = np.random.randint(0, self.region, size=2)
    
#     def set_position(self, agent_id, _position, **kwargs):
#         agent = self.agents[agent_id]
#         if 0 <= _position[0] < self.region and 0 <= _position[1] < self.region:
#             agent.position = _position
    
#     def modify_position(self, agent_id, _mod_value, **kwargs):
#         agent = self.agents[agent_id]
#         new_position = agent.position + _mod_value
#         self.set_position(agent_id, new_position)

# class BattleEnv(AgentBasedSimulation):
#     def __init__(self, **kwargs):
#         self.agents = kwargs['agents']
#         self.position = GridPositionComponent(**kwargs)
    
#     def reset(self, **kwargs):
#         for agent_id in self.agents:
#             self.position.reset(agent_id)
    
#     def step(self, action_dict, **kwargs):
#         for agent_id, action in action_dict.items():
#             self.position.modify_position(agent_id,  action['move'])

#         # Agents can attack other agents from different team
#         pass

#     def render(self, **kwargs):
#         pass

#     def get_obs(self, agent_id, **kwargs):
#         pass

#     def get_reward(self, agent_id, **kwargs):
#         pass

#     def get_done(self, agent_id, **kwargs):
#         pass

#     def get_all_done(self, **kwargs):
#         pass

#     def get_info(self, agent_id, **kwargs):
#         pass

# agents = {f'agent{i}': BattleAgent(id=f'agent{i}', move_range=1, attack_range=2, attack_strength=0.5, starting_position=None) for i in range(20)}
# env = BattleEnv(agents=agents, region=10)
# env.reset()
# env.step({agent.id: agent.action_space.sample() for agent in agents.values()})

# %% Define component to handle the agents' health and process attack action
# Keep track of this in the agent object
import numpy as np

from admiral.envs import AgentBasedSimulation, Agent

# Let's define environment with a state that is made up of:
# 1. Agent positions
# 2. Agent health
# 3. Agent team
# 
# Agents can move around in the environment
# Agents can attack other agents from different team

from gym.spaces import MultiBinary, Box, Dict

class GridPositionAgent(Agent):
    def __init__(self, starting_position=None, **kwargs):
        super().__init__(**kwargs)
        self.starting_position = starting_position

class MovingAndAttackingAgent(Agent):
    def __init__(self, move_range=None, attack_range=None, attack_strength=None, **kwargs):
        super().__init__(**kwargs)
        self.move_range = move_range
        self.attack_range = attack_range
        self.attack_strength = attack_strength
        
        self.action_space = Dict({
            'move': Box(-self.move_range, self.move_range, (2,), np.int),
            'attack': MultiBinary(1),
        })

class HealthAgent(Agent):
    def __init__(self, initial_health=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_health = initial_health

class BattleAgent(GridPositionAgent, MovingAndAttackingAgent, HealthAgent): pass

class GridPositionComponent:
    def __init__(self, agents=None, region=None, **kwargs):
        for agent in agents.values():
            assert isinstance(agent, GridPositionAgent)
        self.agents = agents
        self.region = region
    
    def reset(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        if agent.starting_position is not None:
            agent.position = agent.starting_position
        else:
            agent.position = np.random.randint(0, self.region, size=2)
    
    def set_position(self, agent_id, _position, **kwargs):
        agent = self.agents[agent_id]
        if 0 <= _position[0] < self.region and 0 <= _position[1] < self.region:
            agent.position = _position
    
    def modify_position(self, agent_id, _mod_value, **kwargs):
        agent = self.agents[agent_id]
        new_position = agent.position + _mod_value
        self.set_position(agent_id, new_position)

class HealthComponent:
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
    
    def reset(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        agent.health = agent.initial_health
    
    def set_health(self, agent_id, _health, **kwargs):
        self.agents[agent_id].health = _health
    
    def modify_health(self, agent_id, _mod_value, **kwargs):
        self.set_health(agent_id, self.agents[agent_id].health + _mod_value)

class BattleEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.position = GridPositionComponent(**kwargs)
        self.health = HealthComponent(**kwargs)
    
    def reset(self, **kwargs):
        for agent_id in self.agents:
            self.position.reset(agent_id)
            self.health.reset(agent_id)
    
    def step(self, action_dict, **kwargs):
        for my_id, action in action_dict.items():
            if action['attack']:
                attacking_agent = self.agents[my_id]
                for other_id, other_agent in self.agents.items():
                    if other_id == my_id:
                        # Cannot attack yourself
                        continue
                    elif abs(attacking_agent.position[0] - other_agent.position[0]) > attacking_agent.attack_range or \
                        abs(attacking_agent.position[1] - other_agent.position[1]) > attacking_agent.attack_range:
                        # Agent too far away
                        continue
                    else:
                        self.health.modify_health(other_id, -attacking_agent.attack_strength)
                        self.health.modify_health(my_id, attacking_agent.attack_strength)

        for agent_id, action in action_dict.items():
            self.position.modify_position(agent_id,  action['move'])

        for agent in self.agents.values():
            self.health.modify_health(agent_id, -1) # Some kind of entropy

    def render(self, **kwargs):
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

agents = {f'agent{i}': BattleAgent(id=f'agent{i}', move_range=1, attack_range=2, attack_strength=10, initial_health=100) for i in range(20)}
env = BattleEnv(agents=agents, region=10)
env.reset()
env.step({agent.id: agent.action_space.sample() for agent in agents.values()})

# %%
