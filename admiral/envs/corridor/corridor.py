
from gym.spaces import Discrete
import numpy as np

from admiral.envs import AgentBasedSimulation
from admiral.envs import SimpleAgent as Agent

class Corridor(AgentBasedSimulation):
    """
    Simple Corridor Environment used for testing. A single agent start at position
    0 and can choose to move left, right, or stay still. The agent must learn to 
    move to the right until it reaches the end position. If the agent attempts
    to move left from the start position, it will remain in that start position.

    The agent can observe its own position.
    """
    from enum import IntEnum
    class Actions(IntEnum):
        LEFT = 0
        STAY = 1
        RIGHT = 2

    def __init__(self, config): 
        self.start = 0
        self.end = config['end']
        self.agents = config['agents']
    
    def reset(self, **kwargs):
        self.pos = self.start
    
    def step(self, actions, **kwargs):
        action = actions['agent0']
        if action == self.Actions.LEFT and self.pos != self.start:
            self.pos -= 1
        elif action == self.Actions.RIGHT:
            self.pos += 1
        # else: Don't move
        #   self.pos = self.pos

    def render(self, *args, fig=None, **kwargs):
        """
        Visualize the state of the environment. If a figure is received, then we
        will draw but not actually plot because we assume the caller will do the
        work (e.g. with an Animation object). If there is no figure received, then
        we will draw and plot the environment.
        """
        draw_now = fig is None
        if draw_now:
            from matplotlib import pyplot as plt
            fig = plt.gcf()

        fig.clear()
        ax = fig.gca()
        ax.set(xlim=(-0.5, self.end + 0.5), ylim=(-0.5, 0.5))
        ax.set_xticks(np.arange(-0.5, self.end + 0.5, 1.))
        ax.scatter(self.pos, 0, marker='s', s=200, c='g')
    
        if draw_now:
            plt.plot()
            plt.pause(1e-17)
    
    def get_obs(self, agent_id, **kwargs):
        return self.pos
    
    def get_reward(self, agent_id, **kwargs):
        return 10 if self.pos == self.end else -1
    
    def get_done(self, agent_id, **kwargs):
        return self.pos == self.end
    
    def get_all_done(self, **kwargs):
        return self.pos == self.end
    
    def get_info(self, agent_id, **kwargs):
        return {}
    
    @classmethod
    def build(cls, env_config={}):
        config = {
            'end': 5,
            # agents determined after end is set
        }

        config['end'] = env_config.get('end', config['end'])
        config['agents'] = {'agent0': Agent(
            id='agent0',
            observation_space=Discrete(config['end'] + 1),
            action_space=Discrete(3)
        )}

        return cls(config)
