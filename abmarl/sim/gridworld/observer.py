
from abc import ABC, abstractmethod

import numpy as np

from abmarl.tools import Box
from abmarl.sim.agent_based_simulation import ObservingAgent
from abmarl.sim.gridworld.base import GridWorldBaseComponent
from abmarl.sim.gridworld.agent import GridObservingAgent, AmmoObservingAgent
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


class AbsoluteEncodingObserver(ObserverBaseComponent):
    """
    Observe the agents in the grid according to their actual positions.

    This Observer represents agents by their encoding on cells according to their
    actual positions in the grid.
    If there are multiple agents on a single cell with different encodings, only
    a single randomly chosen encoding will be observed. To be consistent with other
    built-in observers, masked cells are indicated as -2. Typially, -1 is reserved
    for out of bounds encoding, but because this Observer only reports cells in the
    grid, we don't need an out of bounds distinction. Instead, in order for the observing
    agent to identify itself distinctly from other agents of the same encoding,
    it is reported as a -1.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        max_encoding = max([agent.encoding for agent in self.agents.values()])
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.observation_space[self.key] = Box(
                    -2, max_encoding, (self.rows, self.cols), int
                )
                agent.null_observation[self.key] = -2 * np.ones(
                    (self.rows, self.cols), dtype=int
                )

    @property
    def key(self):
        """
        This Observer's key is "absolute_encoding".
        """
        return 'absolute_encoding'

    @property
    def supported_agent_type(self):
        """
        This Observer work with GridObservingAgents
        """
        return GridObservingAgent

    def get_obs(self, agent, **kwargs):
        """
        The agent observes the grid.

        The observation may include the agent itself indicated by a -1, other
        agents indicated by their encodings, empty space indicated with a 0, and
        masked cells indicated as -2, which are masked either because they are
        too far away or because they are blocked from view by view-blocking agents.
        """
        if not isinstance(agent, self.supported_agent_type):
            return {}

        # To generate the observation, we first create a local grid and mask using
        # the agent's view_range. Then we convolve local grid and mask together.
        # Finally, we subsitute a portion of the full grid with the local grid.

        # Generate a local grid and an observation mask
        local_grid, mask = gu.create_grid_and_mask(
            agent, self.grid, agent.view_range, self.agents
        )

        # Convolve the local grid observation with the mask.
        convolved_grid = np.zeros((2 * agent.view_range + 1, 2 * agent.view_range + 1), dtype=int)
        for r in range(2 * agent.view_range + 1):
            for c in range(2 * agent.view_range + 1):
                if mask[r, c]: # We can see this cell
                    candidate_agents = local_grid[r, c]
                    if candidate_agents is None: # This cell is out of bounds
                        continue # Skip this since these cells will be cropped out
                    elif not candidate_agents: # In bounds empty cell
                        convolved_grid[r, c] = 0
                    else: # Observe one of the agents at this cell
                        # Prioritize observing yourself
                        if agent.id in candidate_agents:
                            convolved_grid[r, c] = -1
                        else:
                            convolved_grid[r, c] = np.random.choice([
                                other.encoding
                                for other in candidate_agents.values()
                            ])
                else: # Cell blocked by agent. Indicate invisible with -2
                    convolved_grid[r, c] = -2

        # Substitute the local grid in place in the full grid
        obs = -2 * np.ones((self.rows, self.cols), dtype=int)
        (r, c) = agent.position
        r_lower = max([0, r - agent.view_range])
        r_upper = min([self.grid.rows - 1, r + agent.view_range]) + 1
        c_lower = max([0, c - agent.view_range])
        c_upper = min([self.grid.cols - 1, c + agent.view_range]) + 1
        obs[r_lower:r_upper, c_lower:c_upper] = convolved_grid[
            (r_lower+agent.view_range-r):(r_upper+agent.view_range-r),
            (c_lower+agent.view_range-c):(c_upper+agent.view_range-c)
        ]

        return {self.key: obs}


class PositionCenteredEncodingObserver(ObserverBaseComponent):
    """
    Observe a subset of the grid centered on the agent's position.

    The observation is centered around the observing agent's position. Each agent
    in the "observation window" is recorded in the relative cell using its encoding.
    If there are multiple agents on a single cell with different encodings, the
    agent will observe only one of them chosen at random.
    """
    def __init__(self, observe_self=True, **kwargs):
        super().__init__(**kwargs)
        self.observe_self = observe_self
        max_encoding = max([agent.encoding for agent in self.agents.values()])
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.observation_space[self.key] = Box(
                    -2, max_encoding, (agent.view_range * 2 + 1, agent.view_range * 2 + 1), int
                )
                agent.null_observation[self.key] = -2 * np.ones(
                    (agent.view_range * 2 + 1, agent.view_range * 2 + 1),
                    dtype=int
                )

    @property
    def key(self):
        """
        This Observer's key is "position_centered_encoding".
        """
        return 'position_centered_encoding'

    @property
    def supported_agent_type(self):
        """
        This Observer works with GridObservingAgents.
        """
        return GridObservingAgent

    @property
    def observe_self(self):
        """
        Agents can observe themselves, which may hide important information if
        overlapping is important. This can be turned off by setting observe_self
        to False.
        """
        return self._observe_self

    @observe_self.setter
    def observe_self(self, value):
        assert type(value) is bool, "Observe self must be a boolean."
        self._observe_self = value

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
        obs = np.zeros((2 * agent.view_range + 1, 2 * agent.view_range + 1), dtype=int)
        for r in range(2 * agent.view_range + 1):
            for c in range(2 * agent.view_range + 1):
                if mask[r, c]: # We can see this cell
                    candidate_agents = local_grid[r, c]
                    if candidate_agents is None: # This cell is out of bounds
                        obs[r, c] = -1
                    elif not candidate_agents: # In bounds empty cell
                        obs[r, c] = 0
                    else: # Observe one of the agents at this cell
                        if self.observe_self:
                            obs[r, c] = np.random.choice([
                                other.encoding for other in candidate_agents.values()
                            ])
                        else:
                            choices = [
                                other.encoding
                                for other in candidate_agents.values()
                                if other.id != agent.id
                            ]
                            # It may be that the observing agent is the only agent
                            # at this location but it cannot observe itself, which
                            # makes choices an empty list.
                            obs[r, c] = np.random.choice(choices) if choices else 0
                else: # Cell blocked by agent. Indicate invisible with -2
                    obs[r, c] = -2

        return {self.key: obs}


class StackedPositionCenteredEncodingObserver(ObserverBaseComponent):
    """
    Observe a subset of the grid centered on the agent's position.

    The observation is centered around the observing agent's position. The observing
    agent sees a stack of observations, one for each encoding, where the
    number of agents of each encoding at a cell is given rather than the encoding
    itself. Out of bounds and masked indicators appear in every grid.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.number_of_encodings = max([agent.encoding for agent in self.agents.values()])
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.observation_space[self.key] = Box(
                    -2,
                    len(self.agents),
                    (agent.view_range * 2 + 1, agent.view_range * 2 + 1, self.number_of_encodings),
                    int
                )
                agent.null_observation[self.key] = -2 * np.ones(
                    (agent.view_range * 2 + 1, agent.view_range * 2 + 1, self.number_of_encodings),
                    dtype=int
                )

    @property
    def key(self):
        """
        This Observer's key is "stacked_position_centered_encoding".
        """
        return 'stacked_position_centered_encoding'

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
            dtype=int
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


class AbsolutePositionObserver(ObserverBaseComponent):
    """
    Agents observe their absolute position.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.observation_space[self.key] = Box(
                    np.array([0, 0], dtype=int),
                    np.array([self.grid.rows - 1, self.grid.cols - 1], dtype=int),
                    dtype=int
                )
                agent.null_observation[self.key] = np.zeros((2,), dtype=int)

    @property
    def key(self):
        """
        This Observer's key is "position".
        """
        return 'position'

    @property
    def supported_agent_type(self):
        """
        This Observer works with ObservingAgents
        """
        return ObservingAgent

    def get_obs(self, agent, **kwargs):
        """
        Agents observe their absolute position.
        """
        if not isinstance(agent, self.supported_agent_type):
            return {}
        else:
            return {self.key: agent.position}


class AmmoObserver(ObserverBaseComponent):
    """
    Agents observe their own ammo.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.observation_space[self.key] = Box(
                    0,
                    agent.initial_ammo,
                    shape=(1,),
                    dtype=int
                )
                agent.null_observation[self.key] = 0

    @property
    def key(self):
        """
        This Observer's key is "ammo".
        """
        return 'ammo'

    @property
    def supported_agent_type(self):
        """
        This Observer works with AmmoObservingAgents.
        """
        return AmmoObservingAgent

    def get_obs(self, agent, **kwargs):
        """
        Agents observe their own ammo
        """
        if not isinstance(agent, self.supported_agent_type):
            return {}
        else:
            return {self.key: agent.ammo}
