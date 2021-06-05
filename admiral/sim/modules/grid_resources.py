import numpy as np


class GridResources:
    """
    GridResources provides resources that exist on the grid and can be consumed
    by agents in the simulation via their "harvest" action. The resources will
    replenish over time. The grid is covered up to some coverage percentage, and
    the initial value of the resources on each cell are random between the minimum
    and maximum values.

        max_value: double
            The maximum value that a resource can reach. Default 1.0
        min_value: double
            The minimum value that a resource can reach and still be able
            to regenerate itself over time. If the resource value falls below the
            min_value, then the resource will not revive itself. Default 0.1
        revive_rate: double
            The rate of revival for each of the resources. Default 0.04
        coverage: double
            The ratio of the map that is covered with a resource. Default 0.75.
    """
    def __init__(self, config):
        self.region = config['region']
        self.coverage = config['coverage']
        self.min_value = config['min_value']
        self.max_value = config['max_value']
        self.revive_rate = config['revive_rate']

    def reset(self, **kwargs):
        """
        Reset the grid and cover with resources.
        """
        coverage_filter = np.zeros((self.region, self.region))
        coverage_filter[np.random.uniform(0, 1, (self.region, self.region)) < self.coverage] = 1.
        self.resources = np.multiply(
            np.random.uniform(self.min_value, self.max_value, (self.region, self.region)),
            coverage_filter
        )

    def harvest(self, location, amount, **kwargs):
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
        self.resources[self.resources >= self.min_value] += self.revive_rate
        self.resources[self.resources >= self.max_value] = self.max_value

    def render(self, *args, fig=None, **kwargs):
        draw_now = fig is None
        if draw_now:
            from matplotlib import pyplot as plt
            fig = plt.gcf()
        import seaborn as sns

        fig.clear()
        ax = fig.gca()
        ax = sns.heatmap(np.flipud(self.resources), ax=ax, cmap='Greens')

        if draw_now:
            plt.plot()
            plt.pause(1e-17)

        return ax

    @classmethod
    def build(cls, sim_config={}):
        config = {
            'region': 10,
            'max_value': 1.,
            'min_value': 0.1,
            'revive_rate': 0.04,
            'coverage': 0.75
        }
        for key, value in config.items():
            config[key] = sim_config.get(key, value)
        return cls(config)
