
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

class BattleAgent(Agent):
    def __init__(self, move_range=None, attack_range=None, **kwargs):
        super().__init__(**kwargs)
        self.move_range = move_range
        self.attack_range = attack_range
        
        self.action_space = Dict({
            'move': Box(-self.move_range, self.move_range, (2,), np.int),
            'attack': MultiBinary(1),
        })

class GridPositionComponent:
    def __init__(self, region=None, starting_positions=None, **kwargs):
        self.starting_positions = {agent_id: starting_position for agent_id, starting_position in starting_positions.items()}
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

agents = {f'agent{i}': BattleAgent(id=f'agent{i}', move_range=1, attack_range=2, attack_strength=0.5) for i in range(20)}
starting_positions = {agent_id: np.array([2, 4]) for agent_id in agents}
env = BattleEnv(agents=agents, region=10, starting_positions=starting_positions)
env.reset()
env.step({agent.id: agent.action_space.sample() for agent in agents.values()})

# %%
