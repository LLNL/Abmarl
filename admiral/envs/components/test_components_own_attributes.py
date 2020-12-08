
# %% Define skeleton of environment that implements an ABS interface
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


# %% Define component to handle the agents' positions and any modifications to the position.
# # Keep track of this as a separate attribute in the component
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

# class BattleAgent(Agent):
#     def __init__(self, move_range=None, attack_range=None, **kwargs):
#         super().__init__(**kwargs)
#         self.move_range = move_range
#         self.attack_range = attack_range
        
#         self.action_space = Dict({
#             'move': Box(-self.move_range, self.move_range, (2,), np.int),
#             'attack': MultiBinary(1),
#         })

# class GridPositionComponent:
#     def __init__(self, region=None, starting_positions=None, **kwargs):
#         self.starting_positions = {agent_id: starting_position for agent_id, starting_position in starting_positions.items()}
#         self.region = region
    
#     def reset(self, agent_id, **kwargs):
#         self.agent_positions = {}
#         starting_position = self.starting_position[agent_id]
#         if starting_position is not None:
#             self.agent_positions[agent_id] = starting_position
#         else:
#             self.agent_positions[agent_id] = np.random.randint(0, self.region, size=(2,))
    
#     def set_position(self, agent_id, _position, **kwargs):
#         if 0 <= _position[0] < self.region and 0 <= _position[1] < self.region:
#             self.agent_positions[agent_id] = _position
    
#     def modify_position(self, agent_id, _mod_value, **kwargs):
#         new_position = self.agent_positions[agent_id] + _mod_value
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

# agents = {f'agent{i}': BattleAgent(id=f'agent{i}', move_range=1, attack_range=2, attack_strength=0.5) for i in range(20)}
# starting_positions = {agent_id: np.array([2, 4]) for agent_id in agents}
# env = BattleEnv(agents=agents, region=10, starting_positions=starting_positions)
# env.reset()
# env.step({agent.id: agent.action_space.sample() for agent in agents.values()})




# %% Define component to handle the agents' team, health, and attacking
# Keep track of this as a separate attribute in the component
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

class HealthAgent(Agent):
    def __init__(self, initial_health=None, min_health=0, max_health=100, **kwargs):
        super().__init__(**kwargs)
        self.initial_health = initial_health
        self.min_health = min_health
        self.max_health = max_health

class HealthComponent:
    def __init__(self, agents=None, initial_health=None, **kwargs):
        if initial_health is not None:
            self.initial_health = {agent_id: ih for agent_id, ih in initial_health.items()}
        else:
            self.initial_health = {agent_id: None for agent_id in agents}
    
    def reset(self, agent_id, **kwargs):
        self.agent_health = {}
        init_health = self.initial_health[agent_id]
        if init_health is not None:
            self.agent_health[agent_id] = init_health
        else:
            self.agent_health[agent_id] = np.random.uniform(self.min_health[agent_id], self.max_health[agent_id])
    
    def set_health(self, agent_id, _health, **kwargs):
        agent = self.agents[agent_id]
        if agent.min_health <= _health <= agent.max_health:
            agent.health = _health
    
    def modify_health(self, agent_id, _mod_value, **kwargs):
        self.set_health(agent_id, self.agents[agent_id].health + _mod_value)

class BattleAgent(Agent):
    def __init__(self, move_range=None, attack_range=None, team=None, **kwargs):
        super().__init__(**kwargs)
        self.move_range = move_range
        self.attack_range = attack_range
        
        self.action_space = Dict({
            'move': Box(-self.move_range, self.move_range, (2,), np.int),
            'attack': MultiBinary(1),
        })

class GridPositionComponent:
    def __init__(self, region=None, agents=None, starting_positions=None, **kwargs):
        if starting_positions is not None:
            self.starting_positions = {agent_id: starting_position for agent_id, starting_position in starting_positions.items()}
        else:
            self.starting_positions = {agent_id: None for agent_id in self.agents}
        self.region = region
    
    def reset(self, agent_id, **kwargs):
        self.agent_positions = {}
        starting_position = self.starting_position[agent_id]
        if starting_position is not None:
            self.agent_positions[agent_id] = starting_position
        else:
            self.agent_positions[agent_id] = np.random.randint(0, self.region, size=(2,))
    
    def set_position(self, agent_id, _position, **kwargs):
        if 0 <= _position[0] < self.region and 0 <= _position[1] < self.region:
            self.agent_positions[agent_id] = _position
    
    def modify_position(self, agent_id, _mod_value, **kwargs):
        new_position = self.agent_positions[agent_id] + _mod_value
        self.set_position(agent_id, new_position)

class BattleEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.position = GridPositionComponent(**kwargs)
    
    def reset(self, **kwargs):
        for agent_id in self.agents:
            self.position.reset(agent_id)
    
    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            self.position.modify_position(agent_id,  action['move'])

        # Agents can attack other agents from different team
        pass

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

agents = {f'agent{i}': BattleAgent(id=f'agent{i}', move_range=1, attack_range=2, attack_strength=0.5, team=i%2) for i in range(20)}
starting_positions = {agent_id: np.array([2, 4]) for agent_id in agents}
env = BattleEnv(agents=agents, region=10, starting_positions=starting_positions)
env.reset()
env.step({agent.id: agent.action_space.sample() for agent in agents.values()})

# After some processing, this way reveals itself to be the difference between using
# dictionaries and objects. Here, I have to create a dictionary mapping the id
# to some parameter for every parameter that I care about, such as min_health,
# max_health, initial_health, starting_position, etc. It is definitely easier
# to just keep it in the agent object.

# %%
