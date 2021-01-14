
from matplotlib import pyplot as plt
import numpy as np

from admiral.envs.components.agent import PositionAgent, SpeedAngleAgent
from admiral.envs.components.state import ContinuousPositionState, SpeedAngleState
from admiral.envs.components.actor import SpeedAngleMovementActor
from admiral.envs.components.done import TooCloseDone
from admiral.envs import AgentBasedSimulation
from admiral.tools.matplotlib_utils import mscatter

class BirdAgent(PositionAgent, SpeedAngleAgent): pass

class Flight(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State
        self.position = ContinuousPositionState(**kwargs)
        self.speed_angle = SpeedAngleState(**kwargs)

        # Actor
        self.move = SpeedAngleMovementActor(position=self.position, speed_angle=self.speed_angle, **kwargs)

        # Done
        self.done = TooCloseDone(position=self.position, **kwargs)

        from gym.spaces import MultiBinary
        for agent in self.agents.values():
            agent.observation_space = {'dummy': MultiBinary(1)}
        self.finalize()

    def reset(self, **kwargs):
        self.position.reset(**kwargs)
        self.speed_angle.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        for agent, action in action_dict.items():
            self.move.process_speed_angle_change(self.agents[agent], action['accelerate'][0], action['banking_angle_change'][0], **kwargs)
        
    def render(self, fig=None, **kwargs):
        fig.clear()

        # Draw the resources
        ax = fig.gca()

        # Draw the agents
        ax.set(xlim=(0, self.position.region), ylim=(0, self.position.region))
        ax.set_xticks(np.arange(0, self.position.region, 1))
        ax.set_yticks(np.arange(0, self.position.region, 1))

        agents_x = [agent.position[1] + 0.5 for agent in self.agents.values()]
        agents_y = [self.position.region - 0.5 - agent.position[0] for agent in self.agents.values()]
        mscatter(agents_x, agents_y, ax=ax, m='o', s=100, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)

    def get_obs(self, agent_id, **kwargs):
        pass

    def get_reward(self, agent_id, **kwargs):
        pass

    def get_done(self, agent_id, **kwargs):
        return self.done.get_done(self.agents[agent_id], **kwargs)

    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)

    def get_info(self, agent_id, **kwargs):
        pass

agents = {
    f'bird{i}': BirdAgent(
            id=f'bird{i}', min_speed=0.5, max_speed=1.0, max_acceleration=0.1, \
            max_banking_angle=90, max_banking_angle_change=90, \
            initial_banking_angle=30
        ) for i in range(10)
    }

env = Flight(
    region=20,
    agents=agents,
    collision_distance=1.0,
)
fig = plt.figure()
env.reset()
env.render(fig=fig)

for i in range(50):
    env.step({agent.id: agent.action_space.sample() for agent in agents.values()})
    env.render(fig=fig)
    for agent in agents:
        print(agent, ': ', env.get_done(agent))
    print('\n')

print(env.get_all_done())


