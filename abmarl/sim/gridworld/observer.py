
from abc import ABC, abstractmethod

from gym.spaces import Box
import numpy as np

from abmarl.sim.gridworld import GridWorldBaseComponent, GridObservingAgent
from abmarl.sim.gridworld.state import GridWorldState

class ObserverBaseComponent(GridWorldBaseComponent, ABC):
    """
    Abstract Observer Component base from which all observer components will inherit.
    """
    @abstractmethod
    def get_obs(self, agent, **kwargs):
        """
        Return this agent's observation.
        """
        pass

    @property
    @abstractmethod
    def key(self):
        """
        The key in the observation dictionary.

        All observers in the gridworld framework use dictionary observations.
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

        If an agent is this type, the Observer will add its channel to the
        agent's observation space and will produce observations for this agent.
        """
        pass

class GridObserver(ObserverBaseComponent):
    """
    Observe a subset of the grid centered on the agent's location.
    """
    def __init__(self, grid_state=None, **kwargs):
        super().__init__(**kwargs)
        self.grid_state = grid_state
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.observation_space[self.key] = Box(
                    -np.inf, np.inf, (agent.view_range, agent.view_range), np.int
                )

    @property
    def grid_state(self):
        """
        GridWorldState object that tracks the state of the grid.
        """
        return self._grid_state
    
    @grid_state.setter
    def grid_state(self, value):
        assert isinstance(value, GridWorldState), "Grid State must be a GridState object."
        self._grid_state = value

    @property
    def key(self):
        """
        This observers key is "grid".
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

        The observation contains other agents, empty spaces, out of bounds, and
        masked cells, which can be blocked from view by other view-blocking agents.

        Returns:
            The observation as a dictionary.
        """
        if not isinstance(agent, GridObservingAgent):
            return {}
        
        # Generate a completely empty grid
        local_grid = np.empty((agent.view_range * 2 + 1, agent.view_range * 2 + 1), dtype=object)
        local_grid.fill(-1)

        # Copy the section of the grid around the agent's location
        (r, c) = agent.position
        r_lower = max([0, r - agent.view_range])
        r_upper = min([self.grid_state.rows - 1, r + agent.view_range]) + 1
        c_lower = max([0, c - agent.view_range])
        c_upper = min([self.grid_state.cols - 1, c + agent.view_range]) + 1
        local_grid[
            (r_lower+agent.view_range-r):(r_upper+agent.view_range-r),
            (c_lower+agent.view_range-c):(c_upper+agent.view_range-c)
        ] = self.grid_state.grid[r_lower:r_upper, c_lower:c_upper]

        # Generate an observation mask. The agent's observation can be blocked
        # by walls, which hide the cells "behind" them. In the mask, 1 means that
        # this square is visibile, 0 means that it is invisible.
        mask = np.ones((2 * agent.view_range + 1, 2 * agent.view_range + 1))
        for other in self.agents.values():
            if other.view_blocking:
                r_diff, c_diff = other.position - agent.position
                if -agent.view_range <= r_diff <= agent.view_range and \
                        -agent.view_range <= c_diff <= agent.view_range:
                    if c_diff > 0 and r_diff == 0: # Wall is to the right of agent
                        upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
                        lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
                        for c in range(c_diff, agent.view_range+1):
                            for r in range(-agent.view_range, agent.view_range+1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if lower(c) < r < upper(c):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff > 0 and r_diff > 0: # Wall is below-right of agent
                        upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
                        lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
                        for c in range(c_diff, agent.view_range+1):
                            for r in range(r_diff, agent.view_range+1):
                                if c == c_diff and r == r_diff: continue # Don't mask the wall
                                if lower(c) < r < upper(c):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff == 0 and r_diff > 0: # Wall is below the agent
                        left = lambda t: (c_diff - 0.5) / (r_diff - 0.5) * t
                        right = lambda t: (c_diff + 0.5) / (r_diff - 0.5) * t
                        for c in range(-agent.view_range, agent.view_range+1):
                            for r in range(r_diff, agent.view_range+1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if left(r) < c < right(r):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff < 0 and r_diff > 0: # Wall is below-left of agent
                        upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
                        lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
                        for c in range(c_diff, -agent.view_range-1, -1):
                            for r in range(r_diff, agent.view_range+1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if lower(c) < r < upper(c):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff < 0 and r_diff == 0: # Wall is left of agent
                        upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
                        lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
                        for c in range(c_diff, -agent.view_range-1, -1):
                            for r in range(-agent.view_range, agent.view_range+1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if lower(c) < r < upper(c):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff < 0 and r_diff < 0: # Wall is above-left of agent
                        upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
                        lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
                        for c in range(c_diff, -agent.view_range - 1, -1):
                            for r in range(r_diff, -agent.view_range - 1, -1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if lower(c) < r < upper(c):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff == 0 and r_diff < 0: # Wall is above the agent
                        left = lambda t: (c_diff - 0.5) / (r_diff + 0.5) * t
                        right = lambda t: (c_diff + 0.5) / (r_diff + 0.5) * t
                        for c in range(-agent.view_range, agent.view_range+1):
                            for r in range(r_diff, -agent.view_range - 1, -1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if left(r) < c < right(r):
                                    mask[r + agent.view_range, c + agent.view_range] = 0
                    elif c_diff > 0 and r_diff < 0: # Wall is above-right of agent
                        upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
                        lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
                        for c in range(c_diff, agent.view_range+1):
                            for r in range(r_diff, -agent.view_range - 1, -1):
                                if c == c_diff and r == r_diff: continue # don't mask the wall
                                if lower(c) < r < upper(c):
                                    mask[r + agent.view_range, c + agent.view_range] = 0

        # Convolve the grid observation with the mask.
        obs = np.zeros((2 * agent.view_range + 1, 2 * agent.view_range + 1), dtype=np.int)
        for r in range(2 * agent.view_range + 1):
            for c in range(2 * agent.view_range + 1):
                if mask[r, c]:
                    obj = local_grid[r, c]
                    if obj == -1: # Out of bounds
                        obs[r, c] = -1
                    elif obj is None: # Empty
                        obs[r, c] = 0
                    else: # Something there, so get its encoding
                        obs[r, c] = obj.encoding
                else: # Cell blocked by wall. Indicate invisible with -2
                    obs[r, c] = -2

        return {self.key: obs}
