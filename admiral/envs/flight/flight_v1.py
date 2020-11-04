
from gym.spaces import Box
import numpy as np
# from ray.rllib.env.multi_agent_env import MultiAgentEnv

# from admiral.tools import utils as adu
# from admiral.envs.flight import FlightEnv_v0

class Flight_v1(MultiAgentEnv):
    def __init__(self, config={}):
        self.env = FlightEnv_v0.build(config)
        self.agents = ['bird' + str(i) for i in range(self.env.birds)]

        # The observation that each agent has will be a little differnt from the single agent case.
        # In single agent, every was observed relative to a fixed coordinate. In multiagent,
        # everything is observed relative to the agent in question. So we need to do a bit more
        # math to figure out all the values.
        # We'll do this in a future version of the MA case. For now, we'll just leave all the
        # observations in an absolute reference.
        self.observation_space = self.env.observation_space
        # The action space is only for each agent, so it's actually simpler than the single agent
        # case.
        self.action_space = Box(
            low=np.array([-self.env.acceleration, -self.env.max_relative_angle_change]),
            high=np.array([self.env.acceleration, self.env.max_relative_angle_change]),
            dtype=np.float
        )
    
    def reset(self):
        state = self.env.reset()
        return adu.broadcast(self.agents, state)
    
    def step(self, joint_action):
        accels = [action[0] for action in joint_action.values()]
        angles = [action[1] for action in joint_action.values()]
        action = accels + angles
        state, reward, done, info = self.env.step(action)
        state = adu.broadcast(self.agents, state)
        reward = adu.broadcast(self.agents, reward)
        dones = adu.broadcast(self.agents, done)
        dones['__all__'] = done
        info = adu.broadcast(self.agents, info)
        return state, reward, dones, info
    
    def render(self, *args, **kwargs):
        self.env.render()
    
    def seed(self, value):
        np.random.seed(value)


