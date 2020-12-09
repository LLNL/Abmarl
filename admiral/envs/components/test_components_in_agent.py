
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
#     def __init__(self, move_range=None, attack_range=None, attack_strength=None, **kwargs):
#         super().__init__(**kwargs)
#         self.move_range = move_range
#         self.attack_range = attack_range
#         self.attack_strength = attack_strength
        
#         self.action_space = Dict({
#             'move': Box(-self.move_range, self.move_range, (2,), np.int),
#             'attack': MultiBinary(1),
#         })

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



# class HealthAgent(Agent):
#     def __init__(self, initial_health=None, min_health=0, max_health=100, **kwargs):
#         super().__init__(**kwargs)
#         self.initial_health = initial_health
#         self.min_health = min_health
#         self.max_health = max_health

# class HealthComponent:
#     def __init__(self, agents=None, **kwargs):
#         self.agents = agents
    
#     def reset(self, agent_id, **kwargs):
#         agent = self.agents[agent_id]
#         if agent.initial_health is not None:
#             agent.health = agent.initial_health
#         else:
#             agent.health = np.random.uniform(agent.min_health, agent.max_health)
    
#     def set_health(self, agent_id, _health, **kwargs):
#         agent = self.agents[agent_id]
#         if agent.min_health <= _health <= agent.max_health:
#             agent.health = _health
    
#     def modify_health(self, agent_id, _mod_value, **kwargs):
#         self.set_health(agent_id, self.agents[agent_id].health + _mod_value)



# class BattleAgent(GridPositionAgent, MovingAndAttackingAgent, HealthAgent): pass

# class BattleEnv(AgentBasedSimulation):
#     def __init__(self, **kwargs):
#         self.agents = kwargs['agents']
#         self.position = GridPositionComponent(**kwargs)
#         self.health = HealthComponent(**kwargs)
    
#     def reset(self, **kwargs):
#         for agent_id in self.agents:
#             self.position.reset(agent_id)
#             self.health.reset(agent_id)
    
#     def step(self, action_dict, **kwargs):
#         for my_id, action in action_dict.items():
#             if action['attack']:
#                 attacking_agent = self.agents[my_id]
#                 for other_id, other_agent in self.agents.items():
#                     if other_id == my_id:
#                         # Cannot attack yourself
#                         continue
#                     elif abs(attacking_agent.position[0] - other_agent.position[0]) > attacking_agent.attack_range or \
#                         abs(attacking_agent.position[1] - other_agent.position[1]) > attacking_agent.attack_range:
#                         # Agent too far away
#                         continue
#                     else:
#                         self.health.modify_health(other_id, -attacking_agent.attack_strength)
#                         self.health.modify_health(my_id, attacking_agent.attack_strength)

#         for agent_id, action in action_dict.items():
#             self.position.modify_position(agent_id,  action['move'])

#         for agent in self.agents.values():
#             self.health.modify_health(agent_id, -1) # Some kind of entropy

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

# agents = {f'agent{i}': BattleAgent(id=f'agent{i}', move_range=1, attack_range=2, attack_strength=10) for i in range(20)}
# env = BattleEnv(agents=agents, region=10)
# env.reset()
# env.step({agent.id: agent.action_space.sample() for agent in agents.values()})


# %% Define component to handle the agents' team and process attack action
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



# class HealthAgent(Agent):
#     def __init__(self, initial_health=None, min_health=0, max_health=100, **kwargs):
#         super().__init__(**kwargs)
#         self.initial_health = initial_health
#         self.min_health = min_health
#         self.max_health = max_health

# class HealthComponent:
#     def __init__(self, agents=None, **kwargs):
#         self.agents = agents
    
#     def reset(self, agent_id, **kwargs):
#         agent = self.agents[agent_id]
#         if agent.initial_health is not None:
#             agent.health = agent.initial_health
#         else:
#             agent.health = np.random.uniform(agent.min_health, agent.max_health)
    
#     def set_health(self, agent_id, _health, **kwargs):
#         agent = self.agents[agent_id]
#         if agent.min_health <= _health <= agent.max_health:
#             agent.health = _health
    
#     def modify_health(self, agent_id, _mod_value, **kwargs):
#         self.set_health(agent_id, self.agents[agent_id].health + _mod_value)



# class MovingAndAttackingTeamAgent(Agent):
#     def __init__(self, move_range=None, attack_range=None, attack_strength=None, team=None, **kwargs):
#         super().__init__(**kwargs)
#         self.move_range = move_range
#         self.attack_range = attack_range
#         self.attack_strength = attack_strength
#         self.team = team
        
#         self.action_space = Dict({
#             'move': Box(-self.move_range, self.move_range, (2,), np.int),
#             'attack': MultiBinary(1),
#         })

# class BattleAgent(GridPositionAgent, MovingAndAttackingTeamAgent, HealthAgent): pass

# class BattleEnv(AgentBasedSimulation):
#     def __init__(self, **kwargs):
#         self.agents = kwargs['agents']
#         self.position = GridPositionComponent(**kwargs)
#         self.health = HealthComponent(**kwargs)
    
#     def reset(self, **kwargs):
#         for agent_id in self.agents:
#             self.position.reset(agent_id)
#             self.health.reset(agent_id)
    
#     def step(self, action_dict, **kwargs):
#         for my_id, action in action_dict.items():
#             if action['attack']:
#                 attacking_agent = self.agents[my_id]
#                 for other_id, other_agent in self.agents.items():
#                     if other_id == my_id:
#                         # Cannot attack yourself
#                         continue
#                     elif attacking_agent.team == other_agent.team:
#                         # Cannot attack memebers of your own team
#                         continue
#                     elif abs(attacking_agent.position[0] - other_agent.position[0]) > attacking_agent.attack_range or \
#                         abs(attacking_agent.position[1] - other_agent.position[1]) > attacking_agent.attack_range:
#                         # Agent too far away
#                         continue
#                     else:
#                         self.health.modify_health(other_id, -attacking_agent.attack_strength)
#                         self.health.modify_health(my_id, attacking_agent.attack_strength)

#         for agent_id, action in action_dict.items():
#             self.position.modify_position(agent_id,  action['move'])

#         for agent in self.agents.values():
#             self.health.modify_health(agent_id, -1) # Some kind of entropy

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

# agents = {f'agent{i}': BattleAgent(id=f'agent{i}', move_range=1, attack_range=2, attack_strength=10, team=i%2) for i in range(20)}
# env = BattleEnv(agents=agents, region=10)
# env.reset()
# env.step({agent.id: agent.action_space.sample() for agent in agents.values()})




# %% First attempt to get observation from components. Agents can get their own health and their own position
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

# class GridPositionComponent:
#     def __init__(self, agents=None, region=None, **kwargs):
#         for agent in agents.values():
#             assert isinstance(agent, GridPositionAgent)
#         self.agents = agents
#         self.region = region

#         for agent in agents.values():
#             agent.observation_space['position'] = Box(-region, region, (1,), np.int)
    
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
    
#     def get_obs(self, agent_id, **kwargs):
#         return self.agents[agent_id].position



# class HealthAgent(Agent):
#     def __init__(self, initial_health=None, min_health=0, max_health=100, **kwargs):
#         super().__init__(**kwargs)
#         self.initial_health = initial_health
#         self.min_health = min_health
#         self.max_health = max_health

# class HealthComponent:
#     def __init__(self, agents=None, **kwargs):
#         self.agents = agents
#         for agent in self.agents.values():
#             agent.observation_space['health'] = Box(agent.min_health, agent.max_health, (1,), np.float)
    
#     def reset(self, agent_id, **kwargs):
#         agent = self.agents[agent_id]
#         if agent.initial_health is not None:
#             agent.health = agent.initial_health
#         else:
#             agent.health = np.random.uniform(agent.min_health, agent.max_health)
    
#     def set_health(self, agent_id, _health, **kwargs):
#         agent = self.agents[agent_id]
#         if agent.min_health <= _health <= agent.max_health:
#             agent.health = _health
    
#     def modify_health(self, agent_id, _mod_value, **kwargs):
#         self.set_health(agent_id, self.agents[agent_id].health + _mod_value)
    
#     def get_obs(self, agent_id, **kwargs):
#         return self.agents[agent_id].health



# class MovingAndAttackingTeamAgent(Agent):
#     def __init__(self, move_range=None, attack_range=None, attack_strength=None, team=None, **kwargs):
#         super().__init__(**kwargs)
#         self.move_range = move_range
#         self.attack_range = attack_range
#         self.attack_strength = attack_strength
#         self.team = team
        
#         self.action_space['move'] = Box(-self.move_range, self.move_range, (2,), np.int)
#         self.action_space['attack'] = MultiBinary(1)

# class BattleAgent(GridPositionAgent, MovingAndAttackingTeamAgent, HealthAgent): pass

# class BattleEnv(AgentBasedSimulation):
#     def __init__(self, **kwargs):
#         self.agents = kwargs['agents']
#         self.position = GridPositionComponent(**kwargs)
#         self.health = HealthComponent(**kwargs)

#         self.finalize()
    
#     def reset(self, **kwargs):
#         for agent_id in self.agents:
#             self.position.reset(agent_id)
#             self.health.reset(agent_id)
    
#     def step(self, action_dict, **kwargs):
#         for my_id, action in action_dict.items():
#             if action['attack']:
#                 attacking_agent = self.agents[my_id]
#                 for other_id, other_agent in self.agents.items():
#                     if other_id == my_id:
#                         # Cannot attack yourself
#                         continue
#                     elif attacking_agent.team == other_agent.team:
#                         # Cannot attack memebers of your own team
#                         continue
#                     elif abs(attacking_agent.position[0] - other_agent.position[0]) > attacking_agent.attack_range or \
#                         abs(attacking_agent.position[1] - other_agent.position[1]) > attacking_agent.attack_range:
#                         # Agent too far away
#                         continue
#                     else:
#                         self.health.modify_health(other_id, -attacking_agent.attack_strength)
#                         self.health.modify_health(my_id, attacking_agent.attack_strength)

#         for agent_id, action in action_dict.items():
#             self.position.modify_position(agent_id,  action['move'])

#         for agent in self.agents.values():
#             self.health.modify_health(agent_id, -1) # Some kind of entropy

#     def render(self, **kwargs):
#         pass

#     def get_obs(self, agent_id, **kwargs):
#         return {
#             'position': self.position.get_obs(agent_id),
#             'health': self.health.get_obs(agent_id),
#         }

#     def get_reward(self, agent_id, **kwargs):
#         pass

#     def get_done(self, agent_id, **kwargs):
#         pass

#     def get_all_done(self, **kwargs):
#         pass

#     def get_info(self, agent_id, **kwargs):
#         pass

# agents = {f'agent{i}': BattleAgent(id=f'agent{i}', move_range=1, attack_range=2, attack_strength=10, team=i%2) for i in range(20)}
# env = BattleEnv(agents=agents, region=10)
# env.reset()
# env.get_obs('agent0')
# env.step({agent.id: agent.action_space.sample() for agent in agents.values()})
# env.get_obs('agent0')


# %% First attempt to get observation from components.
# # Agents can get their own health, position, and team.
# # Adding components for movement and attack.
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
#         self.position = None # Will be assigned at reset

# class GridPositionComponent:
#     """
#     Manages the agents' positions.
#     """
#     def __init__(self, agents=None, region=None, **kwargs):
#         for agent in agents.values():
#             assert isinstance(agent, GridPositionAgent)
#         self.agents = agents
#         self.region = region

#         for agent in agents.values():
#             agent.observation_space['position'] = Box(-region, region, (1,), np.int)
    
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
    
#     def get_obs(self, agent_id, **kwargs):
#         return self.agents[agent_id].position



# class MovementAgent(Agent):
#     def __init__(self, move_range=None, **kwargs):
#         super().__init__(**kwargs)
#         self.move_range = move_range

# class MovementActor:
#     """
#     Processes the agents' movement actions into proposed positions.

#     Right now, this doesn't really do anything, but this would be really useful
#     for more complicated movement physics. Currently, this just shows us the software
#     design idea.
#     """
#     def __init__(self, position_component=None, agents=None, **kwargs):
#         self.position = position_component
#         self.agents = agents
#         for agent in agents.values():
#             agent.action_space['move'] = Box(-agent.move_range, agent.move_range, (2,), np.int)
    
#     def process_move(self, agent_id, move, **kwargs):
#         self.position.modify_position(agent_id, move, **kwargs)



# class HealthAgent(Agent):
#     def __init__(self, initial_health=None, min_health=0, max_health=100, **kwargs):
#         super().__init__(**kwargs)
#         self.initial_health = initial_health
#         self.min_health = min_health
#         self.max_health = max_health
#         self.health = None # Will be assigned at reset

# class HealthComponent:
#     def __init__(self, agents=None, **kwargs):
#         self.agents = agents
#         for agent in self.agents.values():
#             agent.observation_space['health'] = Box(agent.min_health, agent.max_health, (1,), np.float)
    
#     def reset(self, agent_id, **kwargs):
#         agent = self.agents[agent_id]
#         if agent.initial_health is not None:
#             agent.health = agent.initial_health
#         else:
#             agent.health = np.random.uniform(agent.min_health, agent.max_health)
    
#     def set_health(self, agent_id, _health, **kwargs):
#         agent = self.agents[agent_id]
#         if agent.min_health <= _health <= agent.max_health:
#             agent.health = _health
    
#     def modify_health(self, agent_id, _mod_value, **kwargs):
#         self.set_health(agent_id, self.agents[agent_id].health + _mod_value)
    
#     def get_obs(self, agent_id, **kwargs):
#         return self.agents[agent_id].health



# class TeamAgent(Agent):
#     def __init__(self, team=None, **kwargs):
#         super().__init__(**kwargs)
#         self.team = team

# class TeamComponent:
#     def __init__(self, agents=None, number_of_teams=None, **kwargs):
#         self.number_of_teams = number_of_teams
#         self.agents = agents

#         for agent in agents.values():
#             agent.observation_space['team'] = Box(0, number_of_teams, (1,), np.int)
    
#     def get_obs(self, agent_id, **kwargs):
#         return self.agents[agent_id].team



# class AttackingAgent(Agent):
#     def __init__(self, attack_range=None, attack_strength=None, **kwargs):
#         super().__init__(**kwargs)
#         self.attack_range = attack_range
#         self.attack_strength = attack_strength

# class AttackActor:
#     def __init__(self, agents=None, health_component=None, **kwargs):
#         self.agents = agents
#         for agent in agents.values():
#             agent.action_space['attack'] = MultiBinary(1)
#         self.health = health_component
    
#     def process_attack(self, attacking_agent_id, attack, **kwargs):
#         if attack:
#             attacking_agent = self.agents[attacking_agent_id]
#             for attacked_agent_id, attacked_agent in self.agents.items():
#                 if attacked_agent_id == attacking_agent_id:
#                     # Cannot attack yourself
#                     continue
#                 elif attacking_agent.team == attacked_agent.team:
#                     # Cannot attack memebers of your own team
#                     continue
#                 elif abs(attacking_agent.position[0] - attacked_agent.position[0]) > attacking_agent.attack_range or \
#                     abs(attacking_agent.position[1] - attacked_agent.position[1]) > attacking_agent.attack_range:
#                     # Agent too far away
#                     continue
#                 else:
#                     self.health.modify_health(attacked_agent_id, -attacking_agent.attack_strength, **kwargs)
#                     self.health.modify_health(attacking_agent_id, attacking_agent.attack_strength, **kwargs)


# class BattleAgent(GridPositionAgent, MovementAgent, AttackingAgent, HealthAgent, TeamAgent): pass

# class BattleEnv(AgentBasedSimulation):
#     def __init__(self, **kwargs):
#         self.agents = kwargs['agents']
#         self.position = GridPositionComponent(**kwargs)
#         self.health = HealthComponent(**kwargs)
#         self.team = TeamComponent(**kwargs)

#         self.move = MovementActor(position_component=self.position, **kwargs)
#         self.attack = AttackActor(health_component=self.health, **kwargs)

#         self.finalize()
    
#     def reset(self, **kwargs):
#         for agent_id in self.agents:
#             self.position.reset(agent_id)
#             self.health.reset(agent_id)
    
#     def step(self, action_dict, **kwargs):
#         for agent_id, action in action_dict.items():
#             self.attack.process_attack(agent_id, action['attack'], **kwargs)

#         for agent_id, action in action_dict.items():
#             self.move.process_move(agent_id, action['move'], **kwargs)

#         for agent in self.agents.values():
#             self.health.modify_health(agent_id, -1) # Some kind of entropy

#     def render(self, **kwargs):
#         pass

#     def get_obs(self, agent_id, **kwargs):
#         return {
#             'position': self.position.get_obs(agent_id),
#             'health': self.health.get_obs(agent_id),
#             'team': self.team.get_obs(agent_id)
#         }

#     def get_reward(self, agent_id, **kwargs):
#         pass

#     def get_done(self, agent_id, **kwargs):
#         pass

#     def get_all_done(self, **kwargs):
#         pass

#     def get_info(self, agent_id, **kwargs):
#         pass

# agents = {f'agent{i}': BattleAgent(id=f'agent{i}', move_range=1, attack_range=2, attack_strength=10, team=i%2) for i in range(20)}
# env = BattleEnv(agents=agents, region=10, number_of_teams=2)
# env.reset()
# env.get_obs('agent0')
# env.step({agent.id: agent.action_space.sample() for agent in agents.values()})
# env.get_obs('agent0')


# %% First attempt to get observation from components.
# # Attempt to create observers that intelligently compile information from the
# # components before passing them on to the agents.
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
#         self.position = None # Will be assigned at reset

# class GridPositionComponent:
#     """
#     Manages the agents' positions.
#     """
#     def __init__(self, agents=None, region=None, **kwargs):
#         self.agents = agents
#         self.region = region

#         # for agent in agents.values():
#         #     agent.observation_space['position'] = Box(-region, region, (1,), np.int)
    
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
    
#     def get_state(self, agent_id, **kwargs):
#         return self.agents[agent_id].position

# class PositionObserver:
#     def __init__(self, position_component=None, agents=None, **kwargs):
#         self.position = position_component
#         self.agents = agents
#         for agent in agents.values():
#             agent.observation_space['position'] = Dict({
#                 agent_id: Box(0, self.position.region, (2,), np.int) for agent_id in agents
#             })
    
#     def get_obs(self, agent_id, **kwargs):
#         return {id: self.position.get_state(id, **kwargs) for id in self.agents}



# class MovementAgent(Agent):
#     def __init__(self, move_range=None, **kwargs):
#         super().__init__(**kwargs)
#         self.move_range = move_range

# class MovementActor:
#     """
#     Processes the agents' movement actions into proposed positions.

#     Right now, this doesn't really do anything, but this would be really useful
#     for more complicated movement physics. Currently, this just shows us the software
#     design idea.
#     """
#     def __init__(self, position_component=None, agents=None, **kwargs):
#         self.position = position_component
#         self.agents = agents
#         for agent in agents.values():
#             agent.action_space['move'] = Box(-agent.move_range, agent.move_range, (2,), np.int)
    
#     def process_move(self, agent_id, move, **kwargs):
#         self.position.modify_position(agent_id, move, **kwargs)



# class HealthAgent(Agent):
#     def __init__(self, initial_health=None, min_health=0, max_health=100, **kwargs):
#         super().__init__(**kwargs)
#         self.initial_health = initial_health
#         self.min_health = min_health
#         self.max_health = max_health
#         self.health = None # Will be assigned at reset

# class HealthComponent:
#     def __init__(self, agents=None, **kwargs):
#         self.agents = agents
    
#     def reset(self, agent_id, **kwargs):
#         agent = self.agents[agent_id]
#         if agent.initial_health is not None:
#             agent.health = agent.initial_health
#         else:
#             agent.health = np.random.uniform(agent.min_health, agent.max_health)
    
#     def set_health(self, agent_id, _health, **kwargs):
#         agent = self.agents[agent_id]
#         if agent.min_health <= _health <= agent.max_health:
#             agent.health = _health
    
#     def modify_health(self, agent_id, _mod_value, **kwargs):
#         self.set_health(agent_id, self.agents[agent_id].health + _mod_value)
    
#     def get_state(self, agent_id, **kwargs):
#         return self.agents[agent_id].health

# class HealthObserver:
#     def __init__(self, health_component=None, agents=None, **kwargs):
#         self.health = health_component
#         self.agents = agents
#         for agent in self.agents.values():
#             agent.observation_space['health'] = Dict({
#                 agent_id: Box(agent.min_health, agent.max_health, (1,), np.float) for agent_id in agents
#             })

#     def get_obs(self, agent_id, **kwargs):
#         return {id: self.health.get_state(id, **kwargs) for id in self.agents}



# class TeamAgent(Agent):
#     def __init__(self, team=None, **kwargs):
#         super().__init__(**kwargs)
#         self.team = team

# class TeamComponent:
#     def __init__(self, agents=None, number_of_teams=None, **kwargs):
#         self.number_of_teams = number_of_teams
#         self.agents = agents
    
#     def get_state(self, agent_id, **kwargs):
#         return self.agents[agent_id].team

# class TeamObserver:
#     def __init__(self, team_component=None, agents=None, **kwargs):
#         self.team = team_component
#         self.agents = agents

#         for agent in agents.values():
#             agent.observation_space['team'] = Dict({
#                 agent_id: Box(0, self.team.number_of_teams, (1,), np.int) for agent_id in agents
#             })
    
#     def get_obs(self, agent_id, **kwargs):
#         return {id: self.team.get_state(id, **kwargs) for id in self.agents}



# class AttackingAgent(Agent):
#     def __init__(self, attack_range=None, attack_strength=None, **kwargs):
#         super().__init__(**kwargs)
#         self.attack_range = attack_range
#         self.attack_strength = attack_strength

# class AttackActor:
#     def __init__(self, agents=None, health_component=None, **kwargs):
#         self.agents = agents
#         for agent in agents.values():
#             agent.action_space['attack'] = MultiBinary(1)
#         self.health = health_component
    
#     def process_attack(self, attacking_agent_id, attack, **kwargs):
#         if attack:
#             attacking_agent = self.agents[attacking_agent_id]
#             for attacked_agent_id, attacked_agent in self.agents.items():
#                 if attacked_agent_id == attacking_agent_id:
#                     # Cannot attack yourself
#                     continue
#                 elif attacking_agent.team == attacked_agent.team:
#                     # Cannot attack memebers of your own team
#                     continue
#                 elif abs(attacking_agent.position[0] - attacked_agent.position[0]) > attacking_agent.attack_range or \
#                     abs(attacking_agent.position[1] - attacked_agent.position[1]) > attacking_agent.attack_range:
#                     # Agent too far away
#                     continue
#                 else:
#                     self.health.modify_health(attacked_agent_id, -attacking_agent.attack_strength, **kwargs)
#                     self.health.modify_health(attacking_agent_id, attacking_agent.attack_strength, **kwargs)



# class BattleAgent(GridPositionAgent, MovementAgent, AttackingAgent, HealthAgent, TeamAgent): pass

# class BattleEnv(AgentBasedSimulation):
#     def __init__(self, **kwargs):
#         self.agents = kwargs['agents']
#         self.position = GridPositionComponent(**kwargs)
#         self.health = HealthComponent(**kwargs)
#         self.team = TeamComponent(**kwargs)

#         self.move = MovementActor(position_component=self.position, **kwargs)
#         self.attack = AttackActor(health_component=self.health, **kwargs)

#         self.health_observer = HealthObserver(health_component=self.health, **kwargs)
#         self.position_observer = PositionObserver(position_component=self.position, **kwargs)
#         self.team_observer = TeamObserver(team_component=self.team, **kwargs)

#         self.finalize()
    
#     def reset(self, **kwargs):
#         for agent_id in self.agents:
#             self.position.reset(agent_id)
#             self.health.reset(agent_id)
    
#     def step(self, action_dict, **kwargs):
#         for agent_id, action in action_dict.items():
#             self.attack.process_attack(agent_id, action['attack'], **kwargs)

#         for agent_id, action in action_dict.items():
#             self.move.process_move(agent_id, action['move'], **kwargs)

#         for agent in self.agents.values():
#             self.health.modify_health(agent_id, -1) # Some kind of entropy

#     def render(self, **kwargs):
#         pass

#     def get_obs(self, agent_id, **kwargs):
#         return {
#             'position': self.position_observer.get_obs(agent_id),
#             'health': self.health_observer.get_obs(agent_id),
#             'team': self.team_observer.get_obs(agent_id)
#         }

#     def get_reward(self, agent_id, **kwargs):
#         pass

#     def get_done(self, agent_id, **kwargs):
#         pass

#     def get_all_done(self, **kwargs):
#         pass

#     def get_info(self, agent_id, **kwargs):
#         pass

# agents = {f'agent{i}': BattleAgent(id=f'agent{i}', move_range=1, attack_range=2, attack_strength=10, team=i%2) for i in range(20)}
# env = BattleEnv(agents=agents, region=10, number_of_teams=2)
# env.reset()
# env.get_obs('agent0')
# env.step({agent.id: agent.action_space.sample() for agent in agents.values()})
# env.get_obs('agent0')


# %% Attempt at get done. If the agent has been attacked 2 more times than it has attacked, then it is done.
# Get all done will be team dependent
import numpy as np

from admiral.envs import AgentBasedSimulation, Agent

# Let's define environment with a state that is made up of:
# 1. Agent positions
# 2. Agent health
# 3. Agent team
# 
# Agents can move around in the environment
# Agents can attack other agents from different team
# 
# Agents are done if they have been attacked twice more than they have attacked.
# Simulation is done if only agents from one team is left

from gym.spaces import MultiBinary, Box, Dict

class GridPositionAgent(Agent):
    def __init__(self, starting_position=None, **kwargs):
        super().__init__(**kwargs)
        self.starting_position = starting_position
        self.position = None # Will be assigned at reset

class GridPositionComponent:
    """
    Manages the agents' positions.
    """
    def __init__(self, agents=None, region=None, **kwargs):
        self.agents = agents
        self.region = region

        # for agent in agents.values():
        #     agent.observation_space['position'] = Box(-region, region, (1,), np.int)
    
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
    
    def get_state(self, agent_id, **kwargs):
        return self.agents[agent_id].position

class PositionObserver:
    def __init__(self, position_component=None, agents=None, **kwargs):
        self.position = position_component
        self.agents = agents
        for agent in agents.values():
            agent.observation_space['position'] = Dict({
                agent_id: Box(0, self.position.region, (2,), np.int) for agent_id in agents
            })
    
    def get_obs(self, agent_id, **kwargs):
        return {id: self.position.get_state(id, **kwargs) for id in self.agents}



class MovementAgent(Agent):
    def __init__(self, move_range=None, **kwargs):
        super().__init__(**kwargs)
        self.move_range = move_range

class MovementActor:
    """
    Processes the agents' movement actions into proposed positions.

    Right now, this doesn't really do anything, but this would be really useful
    for more complicated movement physics. Currently, this just shows us the software
    design idea.
    """
    def __init__(self, position_component=None, agents=None, **kwargs):
        self.position = position_component
        self.agents = agents
        for agent in agents.values():
            agent.action_space['move'] = Box(-agent.move_range, agent.move_range, (2,), np.int)
    
    def process_move(self, agent_id, move, **kwargs):
        self.position.modify_position(agent_id, move, **kwargs)



class HealthAgent(Agent):
    def __init__(self, initial_health=None, min_health=0, max_health=100, **kwargs):
        super().__init__(**kwargs)
        self.initial_health = initial_health
        self.min_health = min_health
        self.max_health = max_health
        self.health = None # Will be assigned at reset

class HealthComponent:
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
    
    def reset(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        if agent.initial_health is not None:
            agent.health = agent.initial_health
        else:
            agent.health = np.random.uniform(agent.min_health, agent.max_health)
    
    def set_health(self, agent_id, _health, **kwargs):
        agent = self.agents[agent_id]
        if agent.min_health <= _health <= agent.max_health:
            agent.health = _health
    
    def modify_health(self, agent_id, _mod_value, **kwargs):
        self.set_health(agent_id, self.agents[agent_id].health + _mod_value)
    
    def get_state(self, agent_id, **kwargs):
        return self.agents[agent_id].health

class HealthObserver:
    def __init__(self, health_component=None, agents=None, **kwargs):
        self.health = health_component
        self.agents = agents
        for agent in self.agents.values():
            agent.observation_space['health'] = Dict({
                agent_id: Box(agent.min_health, agent.max_health, (1,), np.float) for agent_id in agents
            })

    def get_obs(self, agent_id, **kwargs):
        return {id: self.health.get_state(id, **kwargs) for id in self.agents}



class TeamAgent(Agent):
    def __init__(self, team=None, **kwargs):
        super().__init__(**kwargs)
        self.team = team

class TeamComponent:
    def __init__(self, agents=None, number_of_teams=None, **kwargs):
        self.number_of_teams = number_of_teams
        self.agents = agents
    
    def get_state(self, agent_id, **kwargs):
        return self.agents[agent_id].team

class TeamObserver:
    def __init__(self, team_component=None, agents=None, **kwargs):
        self.team = team_component
        self.agents = agents

        for agent in agents.values():
            agent.observation_space['team'] = Dict({
                agent_id: Box(0, self.team.number_of_teams, (1,), np.int) for agent_id in agents
            })
    
    def get_obs(self, agent_id, **kwargs):
        return {id: self.team.get_state(id, **kwargs) for id in self.agents}



class AttackingAgent(Agent):
    def __init__(self, attack_range=None, attack_strength=None, **kwargs):
        super().__init__(**kwargs)
        self.attack_range = attack_range
        self.attack_strength = attack_strength

class AttackedComponent:
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
    
    def reset(self, agent_id, **kwargs):
        self.agents[agent_id].attacked = 0
    
    def set_attacked(self, agent_id, _attacked, **kwargs):
        self.agents[agent_id].attacked = _attacked
    
    def modify_attacked(self, agent_id, value, **kwargs):
        self.set_attacked(agent_id, self.agents[agent_id].attacked + value, **kwargs)
    
    def get_state(self, agent_id, **kwargs):
        return self.agents[agent_id].attacked

class AttackActor:
    def __init__(self, agents=None, health_component=None, attacked_component=None, **kwargs):
        self.agents = agents
        for agent in agents.values():
            agent.action_space['attack'] = MultiBinary(1)
        self.health = health_component
        self.attacked_component = attacked_component
    
    def process_attack(self, attacking_agent_id, attack, **kwargs):
        if attack:
            attacking_agent = self.agents[attacking_agent_id]
            for attacked_agent_id, attacked_agent in self.agents.items():
                if attacked_agent_id == attacking_agent_id:
                    # Cannot attack yourself
                    continue
                elif attacking_agent.team == attacked_agent.team:
                    # Cannot attack memebers of your own team
                    continue
                elif abs(attacking_agent.position[0] - attacked_agent.position[0]) > attacking_agent.attack_range or \
                    abs(attacking_agent.position[1] - attacked_agent.position[1]) > attacking_agent.attack_range:
                    # Agent too far away
                    continue
                else:
                    self.health.modify_health(attacked_agent_id, -attacking_agent.attack_strength, **kwargs)
                    self.health.modify_health(attacking_agent_id, attacking_agent.attack_strength, **kwargs)
                    self.attacked_component.modify_attacked(attacked_agent_id, 1, **kwargs)
                    self.attacked_component.modify_attacked(attacking_agent_id, -1, **kwargs)

class AttackedDoner:
    def __init__(self, attacked_component=None, team_component=None, agents=None, **kwargs):
        self.attacked_component = attacked_component
        self.team = team_component
        self.agents = agents
    
    def get_done(self, agent_id, **kwargs):
        return self.attacked_component.get_state(agent_id) >= 2
    
    def get_all_done(self, **kwargs):
        not_done_by_team = np.zeros(self.team.number_of_teams)
        for agent in self.agents.values():
            if not self.get_done(agent.id):
                not_done_by_team[self.team.get_state(agent.id)] += 1
        return False if sum(not_done_by_team != 0) > 1 else True




class BattleAgent(GridPositionAgent, MovementAgent, AttackingAgent, HealthAgent, TeamAgent): pass

class BattleEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        # Comonents managing the state
        self.agents = kwargs['agents']
        self.position = GridPositionComponent(**kwargs)
        self.health = HealthComponent(**kwargs)
        self.team = TeamComponent(**kwargs)
        self.attacked = AttackedComponent(**kwargs)

        # Actors processing the action signal from agents
        self.move = MovementActor(position_component=self.position, **kwargs)
        self.attack = AttackActor(health_component=self.health, attacked_component=self.attacked, **kwargs)

        # Observers generating observations for agents
        self.health_observer = HealthObserver(health_component=self.health, **kwargs)
        self.position_observer = PositionObserver(position_component=self.position, **kwargs)
        self.team_observer = TeamObserver(team_component=self.team, **kwargs)

        # Doners determinig when an agent is done and when the whole simulation is done
        self.done = AttackedDoner(attacked_component=self.attacked, team_component=self.team, **kwargs)

        self.finalize()
    
    def reset(self, **kwargs):
        for agent_id in self.agents:
            self.position.reset(agent_id)
            self.health.reset(agent_id)
            self.attacked.reset(agent_id)
    
    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            self.attack.process_attack(agent_id, action['attack'], **kwargs)

        for agent_id, action in action_dict.items():
            self.move.process_move(agent_id, action['move'], **kwargs)

        for agent in self.agents.values():
            self.health.modify_health(agent_id, -1) # Some kind of entropy

    def render(self, **kwargs):
        pass

    def get_obs(self, agent_id, **kwargs):
        return {
            'position': self.position_observer.get_obs(agent_id),
            'health': self.health_observer.get_obs(agent_id),
            'team': self.team_observer.get_obs(agent_id)
        }

    def get_reward(self, agent_id, **kwargs):
        pass

    def get_done(self, agent_id, **kwargs):
        return self.done.get_done(agent_id, **kwargs)

    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)

    def get_info(self, agent_id, **kwargs):
        pass

from pprint import pprint
agents = {f'agent{i}': BattleAgent(id=f'agent{i}', move_range=1, attack_range=2, attack_strength=10, team=i%2) for i in range(20)}
env = BattleEnv(agents=agents, region=10, number_of_teams=2)
env.reset()
pprint(env.get_obs('agent0'))
env.step({agent.id: agent.action_space.sample() for agent in agents.values()})
pprint(env.get_obs('agent0'))
pprint(env.get_all_done())

# %%
