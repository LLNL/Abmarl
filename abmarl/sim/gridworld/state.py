
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


class PositionState(StateBaseComponent):
    """
    Manage the agents' positions in the grid.
    """
    def reset(self, **kwargs):
        """
        Give agents their starting positions.

        We use the agent's initial position if it exists. Otherwise, we randomly
        place the agents in the grid.
        """
        max_encoding = max([agent.encoding for agent in self.agents.values()])
        ravelled_available_positions = {
            encoding: set(i for i in range(self.rows * self.cols))
            for encoding in range(1, max_encoding + 1)
        }
        self.grid.reset()
        # First place agents with initial positions.
        for agent in self.agents.values():
            if agent.initial_position is not None:
                r, c = agent.initial_position
                assert self.grid.place(agent, (r, c)), "All initial positions must " + \
                    "be unique or agents with the same initial positions must be overlappable."
                for encoding in range(1, max_encoding + 1):
                    # Remove this cell from any encoding where overlapping is False
                    if encoding not in self.grid.overlapping.get(agent.encoding, {}):
                        try:
                            ravelled_available_positions[encoding].remove(
                                np.ravel_multi_index((r, c), (self.rows, self.cols))
                            )
                        except KeyError:
                            # Catch a key error because this cell might have already
                            # been removed from this encoding
                            continue

        # Now place all the rest of the agents who did not have initial positions
        # and block off those positions as well. We have to do this one agent at
        # a time because the list of available positions is updated after each
        # agent is placed.
        for agent in self.agents.values():
            if agent.initial_position is None:
                n = np.random.choice([*ravelled_available_positions[agent.encoding]], 1)
                r, c = np.unravel_index(n.item(), shape=(self.rows, self.cols))
                assert self.grid.place(agent, (r, c))
                for encoding in range(1, max_encoding + 1):
                    # Remove this cell from any encoding where overlapping is False
                    if encoding not in self.grid.overlapping.get(agent.encoding, {}):
                        try:
                            ravelled_available_positions[encoding].remove(n.item())
                        except KeyError:
                            # Catch a key error because this cell might have already
                            # been removed from this encoding
                            continue


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
