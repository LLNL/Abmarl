
from abc import ABC, abstractmethod

import numpy as np

from admiral.envs import Agent

class ResourceEnv(ABC):
    """
    ResourceEnv contains resources that can regrow over time and that agents can
    harvest.
    
    Provides the process_harvst and regrow api.

    Each resource is valued between the min_value and the max_value, and some coverage
    of the region is covered by this resource. The resources grow back at some
    regrow_rate.

    For replication, you can also provide a premade 2x2 array of resources.

    At reset this environment has populated the world with resources.
    """
    def __init__(self, region=None, coverage=0.75, min_value=0.1, max_value=1.0,
            regrow_rate=0.04, original_resources=None, **kwargs):
        self.original_resources = original_resources
        if self.original_resources is None:
            assert type(region) is int, "Region must be an integer."
            self.region = region
        else:
            self.region = self.original_resources.shape[0]
        self.min_value = min_value
        self.max_value = max_value
        self.regrow_rate = regrow_rate
        self.coverage = coverage
    
    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the resources.
        """
        pass

    @abstractmethod
    def process_harvest(self, location, amount, **kwargs):
        """
        Harvest some amount of resource at this location. Return the amount that
        was actually harvested.
        """
        pass

    @abstractmethod
    def regrow(self, **kwargs):
        """
        Regrow the resources some amount.
        """
        pass

class GridResourceAgent(Agent):
    def __init__(self, max_harvest=None, **kwargs):
        assert max_harvest is not None, "max_harvest must be nonnegative number"
        self.max_harvest = max_harvest
        super().__init__(**kwargs)
    
    @property
    def configured(self):
        return super().configured and self.max_harvest is not None

class GridResourceEnv(ResourceEnv):
    """
    Resources exist in the cells of the grid. The grid is populated with resources
    between the min and max value on some coverage of the region.

    This environment support resource depletion: if a resource falls below the
    minimum value, it will not regrow.
    """
    def __init__(self, agents=None, **kwargs):
        assert type(agents) is dict, "agents must be a dict"
        for agent in agents.values():
            assert isinstance(agent, GridResourceAgent)
        self.agents = agents
        super().__init__(**kwargs)

        from gym.spaces import Box
        for agent in self.agents.values():
            agent.action_space['harvest'] = Box(0, agent.max_harvest, (1,), np.float)
            agent.observation_space['resources'] = None # TODO: Create a resource observation channel

    def reset(self, **kwargs):
        if self.original_resources is not None:
            self.resources = self.original_resources
        else:
            coverage_filter = np.zeros((self.region, self.region))
            coverage_filter[np.random.uniform(0, 1, (self.region, self.region)) < self.coverage] = 1.
            self.resources = np.multiply(
                np.random.uniform(self.min_value, self.max_value, (self.region, self.region)),
                coverage_filter
            )

    def process_harvest(self, location, amount, **kwargs):
        if self.resources[location] - amount >= 0.:
            actual_amount_harvested = amount
        else:
            actual_amount_harvested = self.resources[location]
        self.resources[location] = max([0., self.resources[location] - amount])

        return actual_amount_harvested

    def regrow(self, **kwargs):
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

