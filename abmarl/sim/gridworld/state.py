
from abc import ABC, abstractmethod

import numpy as np

from abmarl.sim.gridworld.base import GridWorldBaseComponent
from abmarl.sim.gridworld.agent import HealthAgent

class StateBaseComponent(GridWorldBaseComponent, ABC):
    """
    Abstract State Component base from which all state components will inherit.
    """

    @abstractmethod
    def reset(self, **kwargs):
        """
        Resets the part of the state for which it is responsible.
        """
        pass

    @abstractmethod
    def update(self, agent, value, **kwargs):
        """
        Update the simulation state.

        Args:
            agent: The agent whose state we will attempt to update.
            value: The proposed value for that agent.
        """
        pass


class PositionState(StateBaseComponent):
    """
    Manage the agent's positions in the grid.

    Every agent occupies a unique cell.
    """
    def reset(self, **kwargs):
        """
        Give agents their starting positions.

        We use the agent's initial position if it exists. Otherwise, we randomly
        place the agent in the grid.
        """
        # Grid lookup by position
        self.grid.fill(None)
        # Prioritize placing agents with initial positions. We must keep track
        # of which positions have been taken so that the random placement below doesn't
        # try to place an agent in an already-taken position.
        ravelled_positions_taken = set()
        for agent in self.agents.values():
            if agent.initial_position is not None:
                r, c = agent.initial_position
                assert self.grid[r, c] is None, f"{agent.id} has the same initial " + \
                    f"position as {self.grid[r, c].id}. All initial positions must be unique."
                agent.position = agent.initial_position
                self.grid[r, c] = agent
                ravelled_positions_taken.add(
                    np.ravel_multi_index(agent.position, (self.rows, self.cols))
                )

        # Now randomly place any other agent who did not come with an initial position.
        ravelled_positions_available = [
            i for i in range(self.rows * self.cols) if i not in ravelled_positions_taken
        ]
        rs, cs = np.unravel_index(
            np.random.choice(ravelled_positions_available, len(self.agents), False),
            shape=(self.rows, self.cols)
        )
        for ndx, agent in enumerate(self.agents.values()):
            if agent.initial_position is None:
                r = rs[ndx]
                c = cs[ndx]
                agent.position = np.array([r, c])
                self.grid[r, c] = agent

    def update(self, agent, new_position, **kwargs):
        """
        Attempt to update the position of the agent.

        If the new_position is inside the grid and not already occupied, then the
        agent will move to that new position.

        Args:
            agent: The agent whose position we attempt to update.
            new_position: The new position. This must be a 2-element numpy array.
        """
        if 0 <= new_position[0] < self.rows and \
                0 <= new_position[1] < self.cols and \
                self.grid[new_position[0], new_position[1]] is None:
            self.grid[agent.position[0], agent.position[1]] = None
            agent.position = new_position
            self.grid[agent.position[0], agent.position[1]] = agent


class HealthState(StateBaseComponent):
    """
    Manage the state of the agents' healths.

    Every HealthAgent has a health. If that health falls to zero, that agent dies
    and is remove from the grid.
    """
    def reset(self, **kwargs):
        """
        Give HealthAgents their starting healths.

        We use the agent's initial health if it exists. Otherwise, we randomly
        assign a value between 0 and 1.
        """
        for agent in self.agents.values():
            if isinstance(agent, HealthAgent):
                if agent.initial_health is not None:
                    agent.health = agent.initial_health
                else:
                    agent.health = np.random.uniform(0, 1)

    def update(self, agent, new_health, **kwargs):
        """
        Attempt to update the agent's health.

        If the health falls to zero, then the agent is removed from the grid.

        Args:
            agent: The agent whose health we attempt to update.
            new_position: The new health.
        """
        if isinstance(agent, HealthAgent):
            agent.health = new_health
            if not agent.active:
                self.grid[agent.position[0], agent.position[1]] = None
                agent.position = None
