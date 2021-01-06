
import numpy as np

from admiral.envs import Agent
from admiral.envs.components.agent import ResourceObservingAgent, HarvestingAgent
from admiral.envs.components.agent import PositionAgent

from admiral.envs.components.state import GridResourceState

class GridResourcesActor:
    """
    Provides the necessary action space for agents who can harvest resources and
    processes the harvesting action.

    resources (ResourceState):
        The resource state handler.

    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, resources=None, agents=None, **kwargs):
        self.resources = resources
        self.agents = agents

        from gym.spaces import Box
        for agent in agents.values():
            if isinstance(agent, HarvestingAgent):
                agent.action_space['harvest'] = Box(0, agent.max_harvest, (1,), np.float)

    def process_harvest(self, agent, amount, **kwargs):
        """
        Harvest some amount of resources at the agent's position.

        agent (HarvestingAgent):
            The agent who has chosen to harvest the resource.

        amount (float):
            The amount of resource the agent wants to harvest.
        
        return (float):
            Return the amount of resources that was actually harvested. This can
            be less than the desired amount if the cell does not have enough resources.
        """
        if isinstance(agent, HarvestingAgent) and isinstance(agent, PositionAgent):
            location = tuple(agent.position)
            resource_before = self.resources.resources[location]
            self.resources.modify_resources(location, -amount)
            return resource_before - self.resources.resources[location]

class GridResourceObserver:
    """
    Agents observe a grid of size resource_view_range centered on their
    position. The values in the grid are the values of the resources in that
    area.

    resources (ResourceState):
        The resource state handler.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, resources=None, agents=None, **kwargs):
        self.resources = resources
        self.agents = agents

        from gym.spaces import Box
        for agent in agents.values():
            if isinstance(agent, ResourceObservingAgent):
                agent.observation_space['resources'] = Box(0, self.resources.max_value, (agent.resource_view_range*2+1, agent.resource_view_range*2+1), np.float)

    def get_obs(self, agent, **kwargs):
        """
        These cells are filled with the values of the resources surrounding the
        agent's position.
        """
        if isinstance(agent, ResourceObservingAgent):
            signal = -np.ones((agent.resource_view_range*2+1, agent.resource_view_range*2+1))

            # Derived by considering each square in the resources as an "agent" and
            # then applied the agent diff logic from above. The resulting for-loop
            # can be written in the below vectorized form.
            (r,c) = agent.position
            r_lower = max([0, r-agent.resource_view_range])
            r_upper = min([self.resources.region-1, r+agent.resource_view_range])+1
            c_lower = max([0, c-agent.resource_view_range])
            c_upper = min([self.resources.region-1, c+agent.resource_view_range])+1
            signal[(r_lower+agent.resource_view_range-r):(r_upper+agent.resource_view_range-r),(c_lower+agent.resource_view_range-c):(c_upper+agent.resource_view_range-c)] = \
                self.resources.resources[r_lower:r_upper, c_lower:c_upper]
            return {'resources': signal}
        else:
            return {}
