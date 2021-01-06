
import numpy as np

from admiral.envs import Agent
from admiral.envs.components.agent import ResourceObservingAgent, HarvestingAgent
from admiral.envs.components.agent import PositionAgent

from admiral.envs.components.state import GridResourceState
from admiral.envs.components.observer import GridResourceObserver

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
