
import numpy as np

from admiral.envs import Agent
from admiral.component_envs.observer import ObservingAgent

class GridResourceHarvestingAgent(Agent):
    def __init__(self, max_harvest=None, **kwargs):
        assert max_harvest is not None, "max_harvest must be nonnegative number"
        self.max_harvest = max_harvest
        super().__init__(**kwargs)
    
    @property
    def configured(self):
        return super().configured and self.max_harvest is not None

class GridResourceComponent:
    """
    Resources exist in the cells of the grid. The grid is populated with resources
    between the min and max value on some coverage of the region.

    This environment support resource depletion: if a resource falls below the
    minimum value, it will not regrow.
    """
    def __init__(self, region=None, agents=None, coverage=0.75, min_value=0.1, max_value=1.0,
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

        assert type(agents) is dict, "agents must be a dict"
        self.agents = agents

        from gym.spaces import Box
        for agent in self.agents.values():
            if isinstance(agent, GridResourceHarvestingAgent):
                agent.action_space['harvest'] = Box(0, agent.max_harvest, (1,), np.float)
            if isinstance(agent, ObservingAgent):
                agent.observation_space['resources'] = Box(0, self.max_value, (agent.view*2+1, agent.view*2+1), np.float)

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

    def act(self, agent, amount, **kwargs):
            location = tuple(agent.position)
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
    
    def get_obs(self, agent_id, **kwargs):
        """
        These cells are filled with the values of the resources surrounding the
        agent.
        """
        agent = self.agents[agent_id]
        if isinstance(agent, ObservingAgent):
            signal = -np.ones((agent.view*2+1, agent.view*2+1))

            # Derived by considering each square in the resources as an "agent" and
            # then applied the agent diff logic from above. The resulting for-loop
            # can be written in the below vectorized form.
            (r,c) = agent.position
            r_lower = max([0, r-agent.view])
            r_upper = min([self.region-1, r+agent.view])+1
            c_lower = max([0, c-agent.view])
            c_upper = min([self.region-1, c+agent.view])+1
            signal[(r_lower+agent.view-r):(r_upper+agent.view-r),(c_lower+agent.view-c):(c_upper+agent.view-c)] = \
                self.resources[r_lower:r_upper, c_lower:c_upper]
            return signal

