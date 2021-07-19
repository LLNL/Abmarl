from enum import IntEnum

from gym.spaces import MultiBinary, Box, Discrete
from matplotlib import pyplot as plt
import numpy as np

from abmarl.sim import AgentBasedSimulation, Agent, PrincipleAgent
from abmarl.tools.matplotlib_utils import mscatter

class ForagingSim(AgentBasedSimulation):
    class Actions(IntEnum):
        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3
        HARVEST = 4

    def __init__(self):
        self.region = 10
        self.agents = {'forager': Agent(
            id='forager',
            action_space=Discrete(5),
            observation_space={
                'food': MultiBinary(5),
                'position': Box(0, self.region, (2,), np.int)
            }
        )}

        self.finalize()

    def reset(self):
        self.food = {}
        for i in range(5):
            self.food[f'food{i}'] = PrincipleAgent(
                id=f'food{i}'
            )
        self.food['food0'].position = np.array([2, 2])
        self.food['food1'].position = np.array([0, 6])
        self.food['food2'].position = np.array([5, 4])
        self.food['food3'].position = np.array([7, 3])
        self.food['food4'].position = np.array([7, 7])
        
        self.agents['forager'].position = np.random.randint(0, self.region, 2)

        self.reward = 0

    def step(self, action_dict):
        forager = self.agents['forager']
        action = action_dict['forager']
        if action == self.Actions.UP:
            if forager.position[0] != 0:
                forager.position += np.array([-1, 0])
                self.reward = -1
            else:
                self.reward = -5
        elif action == self.Actions.RIGHT:
            if forager.position[1] != self.region - 1:
                forager.position += np.array([0, 1])
                self.reward = -1
            else:
                self.reward = -5
        elif action == self.Actions.DOWN:
            if forager.position[0] != self.region - 1:
                forager.position += np.array([1, 0])
                self.reward = -1
            else:
                self.reward = -5
        elif action == self.Actions.LEFT:
            if forager.position[1] != 0:
                forager.position += np.array([0, -1])
                self.reward = -1
            else:
                self.reward = -5
        elif action == self.Actions.HARVEST:
            if self._harvest_food(forager):
                self.reward = 100
            else:
                self.reward = -5

    def get_obs(self, agent_id):
        agent = self.agents[agent_id]
        food = []
        for i in range(5):
            if f'food{i}' in self.food:
                food.append(1)
            else:
                food.append(0)
        return {'food': food, 'position': agent.position}

    def get_reward(self, agent_id):
        reward = self.reward
        self.reward = 0
        return reward

    def get_done(self, agent_id):
        return not self.food

    def get_all_done(self, **kwargs):
        return not self.food

    def get_info(self, agent_id, **kwargs):
        return {}

    def render(self, fig=None, **kwargs):
        fig.clear()

        ax = fig.gca()
        ax.set(xlim=(0, self.region), ylim=(0, self.region))
        ax.set_xticks(np.arange(0, self.region, 1))
        ax.set_yticks(np.arange(0, self.region, 1))
        ax.grid()

        agents_x = [
            agent.position[1] + 0.5 for agent in self.food.values()
        ]
        agents_y = [
            self.region - 0.5 - agent.position[0] for agent in self.food.values()
        ]
        agents_x.append(self.agents['forager'].position[1] + 0.5)
        agents_y.append(self.region - 0.5 - self.agents['forager'].position[0])

        shape = ['s' for _ in self.food]
        shape.append('o')
        color = ['g' for _ in self.food]
        color.append('b')
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, edgecolor='black', facecolor=color)

        plt.plot()
        plt.pause(1e-6)

    def _harvest_food(self, forager):
        for food in self.food.values():
            if np.all(food.position == forager.position):
                self.reward = 100
                del self.food[food.id]
                return True
        return False

if __name__ == '__main__':
    fig = plt.figure()
    sim = ForagingSim()

    for i in range(10):
        print(f"Episode {i}")
        sim.reset()
        print(sim.get_obs('forager'))
        print(sim.get_reward('forager'))
        sim.render(fig=fig)
        x = []
        for j in range(200):
            sim.step({'forager': sim.agents['forager'].action_space.sample()})
            reward = sim.get_reward('forager')
            if reward > 0:
                print(sim.get_obs('forager'))
                print(reward)
            sim.render(fig=fig)
            x = []
