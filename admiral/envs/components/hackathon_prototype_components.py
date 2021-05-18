
import copy

from gym.spaces import Dict, Box
from matplotlib import pyplot as plt
import numpy as np

from admiral.envs import PrincipleAgent

class World:
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.region = region
        self.agents = agents if agents is not None else {}

class ContinuousWorld(World):
    def reset(self, **kwargs):
        for agent in self.agents.values():
            agent.position = np.random.uniform(0, self.region, 2)

class GridWorld(World):
    def reset(self, **kwargs):
        for agent in self.agents.values():
            agent.position = np.random.randint(0, self.region, 2)

    def render(self, fig=None, render_condition=None, **kwargs):
        draw_now = fig is None
        if draw_now:
            fig = plt.gcf()

        ax = fig.gca()
        if render_condition is None:
            agents_x = [agent.position[1] + 0.5 for agent in self.agents.values()]
            agents_y = [self.region - 0.5 - agent.position[0] for agent in self.agents.values()]
        else:
            agents_x = [agent.position[1] + 0.5 for agent in self.agents.values() if render_condition[agent.id]]
            agents_y = [self.region - 0.5 - agent.position[0] for agent in self.agents.values() if render_condition[agent.id]]

        ax.scatter(agents_x, agents_y, marker='o', s=200,  edgecolor='black', facecolor='gray')

        if draw_now:
            plt.plot()
            plt.pause(1e-17)

        return ax

class Movement:
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.region = region
        self.agents = agents if agents is not None else {}

class GridMovement(Movement):
    def process_move(self, agent, direction, **kwargs):
        if 0 <= agent.position[0] + direction[0] < self.region and \
           0 <= agent.position[1] + direction[1] < self.region: # Still inside the boundary, good move
            agent.position += direction 
            return True
        else:
            return False

class Resources:
    def __init__(self, region=None, agents=None, coverage=0.75, min_value=0.1, max_value=1.0, regrow_rate=0.04, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.agents = agents if agents is not None else {}
        self.region = region
        self.min_value = min_value
        self.max_value = max_value
        self.regrow_rate = regrow_rate
        self.coverage = coverage

class GridResources(Resources):
    def reset(self, **kwargs):        
        coverage_filter = np.zeros((self.region, self.region))
        coverage_filter[np.random.uniform(0, 1, (self.region, self.region)) < self.coverage] = 1.
        self.resources = np.multiply(
            np.random.uniform(self.min_value, self.max_value, (self.region, self.region)),
            coverage_filter
        )

    def process_harvest(self, location, amount, **kwargs):
        """
        Process harvesting a certain amount at a certain location. Return the amount
        that was actually harvested here.
        """
        # Process all the harvesting
        if self.resources[location] - amount >= 0.:
            actual_amount_harvested = amount
        else:
            actual_amount_harvested = self.resources[location]
        self.resources[location] = max([0., self.resources[location] - amount])

        return actual_amount_harvested

    def regrow(self, **kwargs):
        """
        Process the regrowth, which is done according to the revival rate.
        """
        self.resources[self.resources >= self.min_value] += self.regrow_rate
        self.resources[self.resources >= self.max_value] = self.max_value

    def render(self, fig=None, **kwargs):
        draw_now = fig is None
        if draw_now:
            from matplotlib import pyplot as plt
            fig = plt.gcf()
        import seaborn as sns

        ax = fig.gca()
        ax = sns.heatmap(np.flipud(self.resources), ax=ax, cmap='Greens')

        if draw_now:
            plt.plot()
            plt.pause(1e-17)

        return ax

class DeathLifeAgent(PrincipleAgent):
    count = 0
    def __init__(self, death=None, life=None, **kwargs):
        super().__init__(**kwargs)
        self.death = death
        self.life = life

        DeathLifeAgent.count += 1

    @classmethod
    def copy(cls, original):
        new_agent = copy.deepcopy(original)
        new_agent.id = f'agent{DeathLifeAgent.count}'
        DeathLifeAgent.count += 1
        return new_agent

    @property
    def configured(self):
        return super().configured and self.death is not None and self.life is not None

class LifeAndDeath:
    def __init__(self, agents=None, **kwargs):
        assert type(agents) is dict, "Agents must be a dict."
        for agent in agents.values():
            assert isinstance(agent, DeathLifeAgent), "Agents must have health in this simulation."
        self.agents = agents
        for agent in self.agents.values():
            agent.is_original = True

    def reset(self, **kwargs):
        for agent_id, agent in list(self.agents.items()):
            if not agent.is_original:
                del self.agents[agent_id]
        DeathLifeAgent.count = len(self.agents)
        for agent in self.agents.values():
            agent.health = np.random.uniform(0.5, 1.0)
            agent.is_alive = True

    def process_health_effects(self, agent, **kwargs):
        if agent.health >= agent.life: # Reproduce
            agent.health /= 2.
            new_agent = DeathLifeAgent.copy(agent)
            new_agent.is_original = False
            self.agents[new_agent.id] = new_agent
        elif agent.health <= agent.death:
            agent.is_alive = False

    def render(self, **kwargs):
        for agent in self.agents.values():
            print(f'{agent.id}: {agent.health}, {agent.is_alive}')

class CompositeEnv:
    def __init__(self, agents=None, **kwargs):
        assert type(agents) is dict, "Agents must be a dict."
        for agent in agents.values():
            assert isinstance(agent, DeathLifeAgent), "Agents must have health in this simulation."
        self.agents = agents
        self.life_and_death = LifeAndDeath(agents=agents, **kwargs)
        self.world = GridWorld(agents=agents, **kwargs)
        self.resources = GridResources(**kwargs)
        self.movement = GridMovement(agents=agents, **kwargs)

    def reset(self, **kwargs):
        self.life_and_death.reset(**kwargs)
        self.world.reset(**kwargs)
        self.resources.reset(**kwargs)

    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.is_alive:
                if 'harvest' in action:
                    amount_harvested = self.resources.process_harvest(tuple(agent.position), action['harvest'])
                    agent.health += amount_harvested
                if 'move' in action:
                    good_move = self.movement.process_move(agent, action['move'])
                    agent.health -= 0.1*sum(action['move']) if good_move else 0.5

                self.life_and_death.process_health_effects(agent)
        self.resources.regrow()

    def render(self, **kwargs):
        fig = plt.gcf()
        fig.clear()
        self.resources.render(fig=fig, **kwargs)
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        self.world.render(fig=fig, render_condition=render_condition, **kwargs)
        plt.plot()
        plt.pause(1e-6)

        self.life_and_death.render(**kwargs)

# --- Use case --- #
region = 10
max_value = 2.0
agents = {f'agent{i}': DeathLifeAgent(
    id=f'agent{i}',
    action_space=Dict({
        'move': Box(-1, 1, (2,), np.int),
        'harvest': Box(0, max_value, (1,), np.float),
    }),
    death=0.,
    life=1.0,
) for i in range(5)}

env = CompositeEnv(
    region=region,
    agents=agents,
    max_value=max_value
)

# TODO: Why do parents and children take EXACT SAME action?
# ANSWER: because the child is a copy of the parent, and that means it copies
# the random number generator, so the actions will be the same.
for ep in range(3):
    print(f'Episode is {ep}')
    env.reset()
    env.render()

    for i in range(24):
        print(i)
        env.step({agent.id: agent.action_space.sample() for agent in agents.values()})
        env.render()

plt.show()