
from abc import ABC, abstractmethod

from gym.spaces import Box
import numpy as np

from abmarl.sim.gridworld.base import GridWorldBaseComponent
from abmarl.sim.gridworld.agent import GridObservingAgent
import abmarl.sim.gridworld.utils as gu


class ObserverBaseComponent(GridWorldBaseComponent, ABC):
    """
    Abstract Observer Component base from which all observer components will inherit.
    """
    @property
    @abstractmethod
    def key(self):
        """
        The key in the observation dictionary.

        The observation space of all observing agents in the gridworld framework is a dict.
        We can build up complex observation spaces with multiple components by
        assigning each component an entry in the observation dictionary. Observations
        will be a dictionary even if your simulation only has one Observer.
        """
        pass

    @property
    @abstractmethod
    def supported_agent_type(self):
        """
        The type of Agent that this Observer works with.

        If an agent is this type, the Observer will add its entry to the
        agent's observation space and will produce observations for this agent.
        """
        pass

    @abstractmethod
    def get_obs(self, agent, **kwargs):
        """
        Observe the state of the simulation.

        Args:
            agent: The agent for which we return an observation.

        Returns:
            This agent's observation.
        """
        pass


class SingleGridObserver(ObserverBaseComponent):
    """
    Observe a subset of the grid centered on the agent's position.

    The observation is centered around the observing agent's position. Each agent
    in the "observation window" is recorded in the relative cell using its encoding.
    If there are multiple agents on a single cell with different encodings, the
    agent will observe only one of them chosen at random.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.observation_space[self.key] = Box(
                    -np.inf, np.inf, (agent.view_range * 2 + 1, agent.view_range * 2 + 1), np.int
                )

    @property
    def key(self):
        """
        This Observer's key is "grid".
        """
        return 'grid'

    @property
    def supported_agent_type(self):
        """
        This Observer works with GridObservingAgents.
        """
        return GridObservingAgent

    def get_obs(self, agent, **kwargs):
        """
        The agent observes a sub-grid centered on its position.

        The observation may include other agents, empty spaces, out of bounds, and
        masked cells, which can be blocked from view by other blocking agents.

        Returns:
            The observation as a dictionary.
        """
        if not isinstance(agent, self.supported_agent_type):
            return {}

        # Generate a local grid and an observation mask
        local_grid, mask = gu.create_grid_and_mask(
            agent, self.grid, agent.view_range, self.agents
        )

        # Convolve the grid observation with the mask.
        obs = np.zeros((2 * agent.view_range + 1, 2 * agent.view_range + 1), dtype=np.int)
        for r in range(2 * agent.view_range + 1):
            for c in range(2 * agent.view_range + 1):
                if mask[r, c]: # We can see this cell
                    candidate_agents = local_grid[r, c]
                    if candidate_agents is None: # This cell is out of bounds
                        obs[r, c] = -1
                    elif not candidate_agents: # In bounds empty cell
                        obs[r, c] = 0
                    else: # Observe one of the agents at this cell
                        obs[r, c] = np.random.choice(
                            [other.encoding for other in candidate_agents.values()]
                        )
                else: # Cell blocked by agent. Indicate invisible with -2
                    obs[r, c] = -2

        return {self.key: obs}


class MultiGridObserver(ObserverBaseComponent):
    """
    Observe a subset of the grid centered on the agent's position.

    The observation is centered around the observing agent's position. The observing
    agent sees a stack of observations, one for each positive encoding, where the
    number of agents of each encoding is given rather than the encoding
    itself. Out of bounds and masked indicators appear in every grid.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.number_of_encodings = 1
        for agent in self.agents.values():
            self.number_of_encodings = max(self.number_of_encodings, agent.encoding)
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.observation_space[self.key] = Box(
                    -2,
                    len(self.agents),
                    (agent.view_range * 2 + 1, agent.view_range * 2 + 1, self.number_of_encodings),
                    np.int
                )

    @property
    def key(self):
        """
        This Observer's key is "grid".
        """
        return 'grid'

    @property
    def supported_agent_type(self):
        """
        This Observer works with GridObservingAgents.
        """
        return GridObservingAgent

    def get_obs(self, agent, **kwargs):
        """
        The agent observes one or more sub-grids centered on its position.

        The observation may include other agents, empty spaces, out of bounds, and
        masked cells, which can be blocked from view by other blocking agents.
        Each grid records the number of agents on a particular cell correlated
        to a specific encoding.

        Returns:
            The observation as a dictionary.
        """
        if not isinstance(agent, self.supported_agent_type):
            return {}

        # Generate a local grid and an observation mask.
        local_grid, mask = gu.create_grid_and_mask(
            agent, self.grid, agent.view_range, self.agents
        )

        # Convolve the grid observation with the mask.
        obs = np.zeros(
            (2 * agent.view_range + 1, 2 * agent.view_range + 1, self.number_of_encodings),
            dtype=np.int
        )
        for encoding in range(self.number_of_encodings):
            for r in range(2 * agent.view_range + 1):
                for c in range(2 * agent.view_range + 1):
                    if mask[r, c]: # We can see this cell
                        candidate_agents = local_grid[r, c]
                        if candidate_agents is None: # This cell is out of bounds
                            obs[r, c, encoding] = -1
                        elif not candidate_agents: # In bounds empty cell
                            obs[r, c, encoding] = 0
                        else: # Observe the number of agents at this cell with this encoding
                            obs[r, c, encoding] = sum([
                                True if other.encoding == encoding + 1 else False
                                for other in candidate_agents.values()
                            ])
                    else: # Cell blocked by agent. Indicate invisible with -2
                        obs[r, c, encoding] = -2

        return {self.key: obs}
