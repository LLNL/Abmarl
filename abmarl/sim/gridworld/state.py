
from abc import ABC, abstractmethod

import numpy as np

from abmarl.sim.gridworld.base import GridWorldBaseComponent, NonOverlappingGrid, OverlappableGrid
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


class UniquePositionState(StateBaseComponent):
    """
    Manage the agent's positions in the grid.

    Every agent occupies a unique cell.
    """
    @StateBaseComponent.grid.setter
    def grid(self, value):
        super(UniquePositionState, type(self)).grid.fset(self, value)
        assert isinstance(value, NonOverlappingGrid), "The grid must be a NonOverlappingGrid object."
        self._grid = value

    def reset(self, **kwargs):
        """
        Give agents their starting positions.

        We use the agent's initial position if it exists. Otherwise, we randomly
        place the agent at unique cells in the grid.
        """
        # Grid lookup by position
        self.grid.reset()
        # Prioritize placing agents with initial positions. We must keep track
        # of which positions have been taken so that the random placement below doesn't
        # try to place an agent in an already-taken position.
        ravelled_positions_taken = set()
        for agent in self.agents.values():
            if agent.initial_position is not None:
                r, c = agent.initial_position
                assert self.grid.query(agent, (r, c)), "All initial positions must be unique."
                self.grid.place(agent, (r, c))
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
                self.grid.place(agent, (r, c))


class OverlappingPositionState(StateBaseComponent):
    """
    Manages the agents' positions in the grid.

    Multiple agents can occupy the same cell.
    """
    @StateBaseComponent.grid.setter
    def grid(self, value):
        super(OverlappingPositionState, type(self)).grid.fset(self, value)
        assert isinstance(value, OverlappableGrid), "The grid must be a Overlappable grid."
        self._grid = value

    def reset(self, **kwargs):
        """
        Give agents their starting positions.

        Use the agent's initial position if specified, otherwise randomly place
        the agents in the grid.
        """
        self.grid.reset()
        # Prioritize placing agents with initial positions. We must keep track
        # of which positions have been taken by non-overlappable agents so that
        # the random placement below doesn't try to place an agent there.
        ravelled_positions_available = set(i for i in range(self.rows * self.cols))
        for agent in self.agents.values():
            if agent.initial_position is not None:
                r, c = agent.initial_position
                assert self.grid.query(agent, (r, c)), "All initial positions must " + \
                    "be unique or agents with same initial positions must be overlappable."
                self.grid.place(agent, (r, c))
                if not agent.overlappable:
                    ravelled_positions_available.remove(
                        np.ravel_multi_index(agent.position, (self.rows, self.cols))
                    )

        # Now place all the non-overlappable agents who did not have initial positions
        # and block off those positions as well. We have to do this one agent at
        # a time because the list of available positions is updated after each
        # agent is placed.
        for agent in self.agents.values():
            if agent.initial_position is None and not agent.overlappable:
                n = np.random.choice([*ravelled_positions_available], 1)
                r, c = np.unravel_index(n, shape=(self.rows, self.cols))
                self.grid.place(agent, (r, c))
                ravelled_positions_available.remove(n)

        # Now place all remaining agents randomly in the available positions
        rs, cs = np.unravel_index(
            np.random.choice([*ravelled_positions_available], len(self.agents), True),
            shape=(self.rows, self.cols)
        )
        for ndx, agent in enumerate(self.agents.values()):
            if agent.initial_position is None and agent.overlappable:
                r = rs[ndx]
                c = cs[ndx]
                self.grid.place(agent, (r, c))


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
