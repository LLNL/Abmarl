
from abc import ABC, abstractmethod

import numpy as np

from abmarl.sim.gridworld.base import GridWorldBaseComponent, GridWorldAgent
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
    @property
    def ravelled_positions_available(self):
        """
        A dictionary mapping the enodings to a set of positions available to
        agents of that encoding at reset. The set should contain cells represented
        in their ravelled form.
        """
        return self._ravelled_positions_available

    @ravelled_positions_available.setter
    def ravelled_positions_available(self, value):
        assert type(value) is dict, "Ravelled Positions available must be a dictionary."
        for encoding, ravelled_positions_available in value.items():
            assert type(encoding) is int, "Ravelled Position keys must be integers."
            assert type(ravelled_positions_available) is set, \
                "Ravelled Position values must be sets."
            for ravelled_cell in ravelled_positions_available:
                assert type(ravelled_cell) is int, "Available cells must be integers. " \
                    "They should be the ravelled presentation of the cell."
        self._ravelled_positions_available = value

    def reset(self, **kwargs):
        """
        Give agents their starting positions.

        We use the agent's initial position if it exists. Otherwise, we randomly
        place the agents in the grid.
        """
        self.grid.reset()

        # Build set of available positions
        self._build_available_positions()

        # Place agents with initial positions.
        for agent in self.agents.values():
            if agent.initial_position is not None:
                self._place_initial_position_agent(agent)

        # Now place agents with variable positions
        for agent in self.agents.values():
            if agent.initial_position is None:
                self._place_variable_position_agent(agent)

    def _build_available_positions(self, **kwargs):
        """
        Define the positions that are available per encoding.
        """
        max_encoding = max([agent.encoding for agent in self.agents.values()])
        self.ravelled_positions_available = {
            encoding: set(i for i in range(self.rows * self.cols))
            for encoding in range(1, max_encoding + 1)
        }

    def _update_available_positions(self, agent_just_placed, **kwargs):
        """
        Update the available positions based on the agent that was just placed.
        """
        for encoding, positions_available in self.ravelled_positions_available.items():
            # Remove this cell from any encoding where overlapping is False
            if encoding not in self.grid.overlapping.get(agent_just_placed.encoding, {}):
                try:
                    positions_available.remove(
                        np.ravel_multi_index(agent_just_placed.position, (self.rows, self.cols))
                    )
                except KeyError:
                    # Catch a key error because this cell might have already
                    # been removed from this encoding
                    continue

    def _place_initial_position_agent(self, ip_agent_to_place, **kwargs):
        """
        Place an agent with an initial position.
        """
        ravelled_initial_position = np.ravel_multi_index(
            ip_agent_to_place.initial_position,
            (self.rows, self.cols)
        )
        assert ravelled_initial_position in \
            self.ravelled_positions_available[ip_agent_to_place.encoding], \
            f"Cell {ip_agent_to_place.initial_position} is not available for " \
            f"{ip_agent_to_place.id}."
        assert self.grid.place(ip_agent_to_place, ip_agent_to_place.initial_position)
        self._update_available_positions(ip_agent_to_place)

    def _place_variable_position_agent(self, var_agent_to_place, **kwargs):
        """
        Place an agent with a variable position.

        This implementation randomly places the agent in one of its available cells.
        """
        try:
            ravelled_position = np.random.choice(
                [*self.ravelled_positions_available[var_agent_to_place.encoding]], 1)
        except ValueError:
            raise RuntimeError(f"Could not find a cell for {var_agent_to_place.id}") from None
        else:
            r, c = np.unravel_index(ravelled_position.item(), shape=(self.rows, self.cols))
            assert self.grid.place(var_agent_to_place, (r, c))
            self._update_available_positions(var_agent_to_place)


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
