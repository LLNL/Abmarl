
from abc import ABC, abstractmethod
from copy import deepcopy
import random

import numpy as np

from abmarl.sim.gridworld.base import GridWorldBaseComponent, GridWorldAgent
from abmarl.sim.gridworld.agent import HealthAgent, AmmoAgent, OrientationAgent
import abmarl.sim.gridworld.utils as gu


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
    def __init__(self, no_overlap_at_reset=False, randomize_placement_order=False, **kwargs):
        super().__init__(**kwargs)
        self.no_overlap_at_reset = no_overlap_at_reset
        self.randomize_placement_order = randomize_placement_order

    @property
    def ravelled_positions_available(self):
        """
        A dictionary mapping the enodings to a list of positions available to
        agents of that encoding at reset. The list should contain cells represented
        in their ravelled form.
        """
        return self._ravelled_positions_available

    @ravelled_positions_available.setter
    def ravelled_positions_available(self, value):
        assert type(value) is dict, "Ravelled Positions available must be a dictionary."
        for encoding, ravelled_positions_available in value.items():
            assert type(encoding) is int, "Ravelled Position keys must be integers."
            assert type(ravelled_positions_available) is list, \
                "Ravelled Position values must be lists."
            for ravelled_cell in ravelled_positions_available:
                assert type(ravelled_cell) in [int, np.int64], "Available cells must be " \
                    "integers. They should be the ravelled presentation of the cell."
        self._ravelled_positions_available = value

    @property
    def no_overlap_at_reset(self):
        """
        Attempt to place each agent on its own cell.

        Agents with initial positions will override this property.
        """
        return self._no_overlap_at_reset

    @no_overlap_at_reset.setter
    def no_overlap_at_reset(self, value):
        assert type(value) is bool, "No overlap at reset must be a boolean."
        self._no_overlap_at_reset = value

    @property
    def randomize_placement_order(self):
        """
        Randomize the order in which each agent in a category is placed.

        All agents with initial positions will still be placed before agents without
        initial positions. Now, the subset of agents with initial positions will
        be placed in random order. Likewise, the subset of agents without initial
        positions will be placed in random order.

        Agents are reshuffled every episode.
        """
        return self._randomize_placement_order

    @randomize_placement_order.setter
    def randomize_placement_order(self, value):
        assert type(value) is bool, "Randomize placement order must be True or False."
        self._randomize_placement_order = value

    def reset(self, **kwargs):
        """
        Give agents their starting positions.

        We use the agent's initial position if it exists. Otherwise, we randomly
        place the agents in the grid.
        """
        self.grid.reset()

        # Shuffle agents if requested
        if self.randomize_placement_order:
            agents = list(self.agents.items())
            random.shuffle(agents)
            self.agents = dict(agents)

        # Build lists of available positions
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
            encoding: [i for i in range(self.rows * self.cols)]
            for encoding in range(1, max_encoding + 1)
        }

    def _update_available_positions(self, agent_just_placed, **kwargs):
        """
        Update the available positions based on the agent that was just placed.
        """
        for encoding, positions_available in self.ravelled_positions_available.items():
            # Remove this cell from any encoding where overlapping is False
            if self.no_overlap_at_reset or \
                    encoding not in self.grid.overlapping.get(agent_just_placed.encoding, {}):
                try:
                    positions_available.remove(
                        np.ravel_multi_index(agent_just_placed.position, (self.rows, self.cols))
                    )
                except ValueError:
                    # Catch a value error because this cell might have already
                    # been removed from this encoding
                    continue

    def _place_initial_position_agent(self, ip_agent_to_place, **kwargs):
        """
        Place an agent with an initial position.
        """
        assert self.grid.place(ip_agent_to_place, ip_agent_to_place.initial_position), \
            f"Cell {ip_agent_to_place.initial_position} is not available for " \
            f"{ip_agent_to_place.id}."
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


class TargetBarriersFreePlacementState(PositionState):
    """
    Place agents in the grid based on relationship to the target.

    Place a target agent, either randomly or based on its initial position. Barrier
    agents can be placed near the target, and free agents can be placed far away
    from the target.

    Note: Agents with initial positions may conflict with the target agent. If
    the target agent is configured for random placement, then we recommend not
    assigning an initial position to any agent.

    Args:
        target_agent: Barrier will cluster near this agent.
        barrier_encodings: Set of encodings indicating which agents are to be treated as barriers.
        free_encodings: Set of encodings indicating which agents are to be treated as free.
        cluster_barriers: Prioritize the placement of barriers near the target.
        scatter_free_agents: Prioritize the placement of free agents away from
            the target.
    """
    def __init__(self,
                 target_agent=None,
                 barrier_encodings=None,
                 free_encodings=None,
                 cluster_barriers=False,
                 scatter_free_agents=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_agent = target_agent
        self.barrier_encodings = barrier_encodings
        self.free_encodings = free_encodings
        self.cluster_barriers = cluster_barriers
        self.scatter_free_agents = scatter_free_agents

    @property
    def target_agent(self):
        """
        The target agent's position is used to place the other agents.
        """
        return self._target_agent

    @target_agent.setter
    def target_agent(self, value):
        if type(value) is str:
            assert value in self.agents, "The target agent must be an agent in the simulation."
            value = self.agents[value]
        else:
            assert value in self.agents.values(), \
                "The target agent must be an agent in the simulation."
        assert isinstance(value, GridWorldAgent), "Target agent must be a GridWorld agent."
        self._target_agent = value

    @property
    def barrier_encodings(self):
        """
        A set of encodings corresponding to the maze's barrier cells.
        """
        return self._barrier_encodings

    @barrier_encodings.setter
    def barrier_encodings(self, value):
        if value is not None:
            assert type(value) is set, "Barrier encodings must be a set."
            for encoding in value:
                assert type(encoding) is int, "Each barrier encoding must be an integer."
            self._barrier_encodings = value
        else:
            self._barrier_encodings = set()

    @property
    def free_encodings(self):
        """
        A set of encodings corresponding to the maze's free cells.
        """
        return self._free_encodings

    @free_encodings.setter
    def free_encodings(self, value):
        if value is not None:
            assert type(value) is set, "Free encodings must be a set."
            for encoding in value:
                assert type(encoding) is int, "Each free encoding must be an integer."
            self._free_encodings = value
        else:
            self._free_encodings = set()

    @property
    def cluster_barriers(self):
        """
        If True, then prioritize placing barriers near the target agent.
        """
        return self._cluster_barriers

    @cluster_barriers.setter
    def cluster_barriers(self, value):
        assert type(value) is bool, "Cluster barriers must be a boolean."
        self._cluster_barriers = value

    @property
    def scatter_free_agents(self):
        """
        If True, then prioritize placing free agents away from the target agent.
        """
        return self._scatter_free_agents

    @scatter_free_agents.setter
    def scatter_free_agents(self, value):
        assert type(value) is bool, "Scatter free agents must be a boolean."
        self._scatter_free_agents = value

    def reset(self, **kwargs):
        """
        Give the agents their starting positions.
        """
        self.grid.reset()

        # Shuffle agents if requested
        if self.randomize_placement_order:
            agents = list(self.agents.items())
            random.shuffle(agents)
            self.agents = dict(agents)

        # Assert that all encodings are captured
        for agent in self.agents.values():
            assert agent.encoding in {*self.barrier_encodings, *self.free_encodings}, \
                "All agent encodings must be either barrier or free cell."

        # Build lists of available encodings
        self._build_available_positions()

        # Manually place target agent at the maze start
        assert self.grid.place(self.target_agent, self._target_start), \
            "Unable to place target agent."
        self._update_available_positions(self.target_agent)

        # Place agents with initial positions.
        for agent in self.agents.values():
            if agent == self.target_agent:
                continue
            if agent.initial_position is not None:
                self._place_initial_position_agent(agent)

        # Now place barrier + free agents with variable positions
        for agent in self.agents.values():
            if agent == self.target_agent:
                continue
            if agent.initial_position is None:
                self._place_variable_position_agent(agent)

    def _build_available_positions(self, **kwargs):
        """
        Define the available positions per encoding.

        Available cells are ordered based on clustering and scattering properties.
        """
        if self.target_agent.initial_position is not None:
            self._target_start = self.target_agent.initial_position
        else:
            self._target_start = np.random.randint(0, (self.rows, self.cols))

        ravelled_barrier_positions = [i for i in range(self.rows * self.cols)]
        if self.cluster_barriers:
            # We sort the available positions according to their distance from target
            # in reverse order becuase we will grab the last position in the list
            # when selecting from the available cells, which will give us the closest
            # cells first.
            ravelled_barrier_positions.sort(
                key=lambda x: np.linalg.norm(
                    np.array([np.unravel_index(x, (self. rows, self.cols))]) - self._target_start
                ),
                reverse=True
            )

        ravelled_free_positions = [i for i in range(self.rows * self.cols)]
        if self.scatter_free_agents:
            # We sort the available positions according to their distance from target
            # becuase we will grab the last position in the list when selecting
            # from the available cells, which will give us the furthest cells first.
            ravelled_free_positions.sort(
                key=lambda x: np.linalg.norm(
                    np.array([np.unravel_index(x, (self. rows, self.cols))]) - self._target_start
                )
            )

        self.ravelled_positions_available = {
            **{
                encoding: deepcopy(ravelled_barrier_positions)
                for encoding in self.barrier_encodings
            },
            **{encoding: deepcopy(ravelled_free_positions) for encoding in self.free_encodings}
        }

    def _place_variable_position_agent(self, var_agent_to_place, **kwargs):
        """
        Place an agent with a variable position.

        Barriers agents will be clustered around the target if cluster_barriers
        is True. Free agents will be scattered far from the target if scatter_free_agents
        is True.
        """
        if (var_agent_to_place.encoding in self.barrier_encodings and self.cluster_barriers) \
            or (var_agent_to_place.encoding in self.free_encodings and
                self.scatter_free_agents):
            try:
                ravelled_position = \
                    self.ravelled_positions_available[var_agent_to_place.encoding][-1]
            except IndexError:
                raise RuntimeError(f"Could not find a cell for {var_agent_to_place.id}") from None
            else:
                r, c = np.unravel_index(ravelled_position, shape=(self.rows, self.cols))
                assert self.grid.place(var_agent_to_place, (r, c))
                self._update_available_positions(var_agent_to_place)
        else:
            super()._place_variable_position_agent(var_agent_to_place)


class MazePlacementState(PositionState):
    """
    Place agents in the grid based on a maze generated around a target.

    Partition the cells into two categories, either a free cell or a barrier, based
    on a maze, which is generated starting at a target agent's position. Specify
    available positions as follows: barrier-encoded agents will be placed at the
    maze barriers, free-encoded agents will be placed at free positions.

    Note: Because the maze is randomly generated at the beginning of each episode
    and because the agents must be placed in either a free cell or barrier cell
    according to their encodings, it is highly recommended that none of your agents
    be given initial positions, except for the target agent.

    Args:
        target_agent: Start the maze generation at this agent's position and place
            the target agent here.
        barrier_encodings: A set of encodings corresponding to the maze's barrier cells.
        free_encodings: A set of encodings corresponding to the maze's free cells.
        cluster_barriers: Prioritize the placement of barriers near the target.
        scatter_free_agents: Prioritize the placement of free agents away from
            the target.
    """
    def __init__(self,
                 target_agent=None,
                 barrier_encodings=None,
                 free_encodings=None,
                 cluster_barriers=False,
                 scatter_free_agents=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_agent = target_agent
        self.barrier_encodings = barrier_encodings
        self.free_encodings = free_encodings
        self.cluster_barriers = cluster_barriers
        self.scatter_free_agents = scatter_free_agents

    @property
    def target_agent(self):
        """
        The target agent is the place from which to start the maze generation.

        Other agents are placed relative to the target.
        """
        return self._target_agent

    @target_agent.setter
    def target_agent(self, value):
        if type(value) is str:
            assert value in self.agents, "The target agent must be an agent in the simulation."
            value = self.agents[value]
        else:
            assert value in self.agents.values(), \
                "The target agent must be an agent in the simulation."
        assert isinstance(value, GridWorldAgent), "Target agent must be a GridWorld agent."
        self._target_agent = value

    @property
    def barrier_encodings(self):
        """
        A set of encodings corresponding to the maze's barrier cells.
        """
        return self._barrier_encodings

    @barrier_encodings.setter
    def barrier_encodings(self, value):
        if value is not None:
            assert type(value) is set, "Barrier encodings must be a set."
            for encoding in value:
                assert type(encoding) is int, "Each barrier encoding must be an integer."
            self._barrier_encodings = value
        else:
            self._barrier_encodings = set()

    @property
    def free_encodings(self):
        """
        A set of encodings corresponding to the maze's free cells.
        """
        return self._free_encodings

    @free_encodings.setter
    def free_encodings(self, value):
        if value is not None:
            assert type(value) is set, "Free encodings must be a set."
            for encoding in value:
                assert type(encoding) is int, "Each free encoding must be an integer."
            self._free_encodings = value
        else:
            self._free_encodings = set()

    @property
    def cluster_barriers(self):
        """
        If True, then prioritize placing barriers near the target agent.
        """
        return self._cluster_barriers

    @cluster_barriers.setter
    def cluster_barriers(self, value):
        assert type(value) is bool, "Cluster barriers must be a boolean."
        self._cluster_barriers = value

    @property
    def scatter_free_agents(self):
        """
        If True, then prioritize placing free agents away from the target agent.
        """
        return self._scatter_free_agents

    @scatter_free_agents.setter
    def scatter_free_agents(self, value):
        assert type(value) is bool, "Scatter free agents must be a boolean."
        self._scatter_free_agents = value

    def reset(self, **kwargs):
        """
        Give the agents their starting positions.
        """
        self.grid.reset()

        # Shuffle agents if requested
        if self.randomize_placement_order:
            agents = list(self.agents.items())
            random.shuffle(agents)
            self.agents = dict(agents)

        # Assert that all encodings are captured
        for agent in self.agents.values():
            assert agent.encoding in {*self.barrier_encodings, *self.free_encodings}, \
                "All agent encodings must be either barrier or free cell."

        # Build lists of available encodings
        self._build_available_positions()

        # Manually place target agent at the maze start
        assert self.grid.place(self.target_agent, self._maze_start), "Unable to place target agent."
        self._update_available_positions(self.target_agent)

        # Place agents with initial positions.
        for agent in self.agents.values():
            if agent == self.target_agent:
                continue
            if agent.initial_position is not None:
                self._place_initial_position_agent(agent)

        # Now place barrier + free agents with variable positions
        for agent in self.agents.values():
            if agent == self.target_agent:
                continue
            if agent.initial_position is None:
                self._place_variable_position_agent(agent)

    def _build_available_positions(self, **kwargs):
        """
        Define the positions available per encoding.

        The avaiable positions is based on maze generated starting from the target's
        position. This maze partitions the cells into two categories: free and barrier.
        If an agent has a barrier encoding, then it can only be placed at a barrier
        cell. If an agent has a free encoding, then it can only be placed at a free
        cell.
        """
        if self.target_agent.initial_position is not None:
            self._maze_start = self.target_agent.initial_position
        else:
            self._maze_start = np.random.randint(0, (self.rows, self.cols))
        maze = gu.generate_maze(self.rows, self.cols, self._maze_start)

        ravelled_barrier_positions = []
        barrier_positions = np.where(maze == 1)
        for ndx in range(len(barrier_positions[0])):
            r, c = barrier_positions[0][ndx], barrier_positions[1][ndx]
            n = np.ravel_multi_index((r, c), (self.rows, self.cols))
            ravelled_barrier_positions.append(n)
        if self.cluster_barriers:
            # We sort the available positions according to their distance from target
            # in reverse order becuase we will grab the last position in the list
            # when selecting from the available cells, which will give us the closest
            # cells first.
            ravelled_barrier_positions.sort(
                key=lambda x: np.linalg.norm(
                    np.array([np.unravel_index(x, (self. rows, self.cols))]) - self._maze_start
                ),
                reverse=True
            )

        ravelled_free_positions = []
        free_positions = np.where(maze == 0)
        for ndx in range(len(free_positions[0])):
            r, c = free_positions[0][ndx], free_positions[1][ndx]
            n = np.ravel_multi_index((r, c), (self.rows, self.cols))
            ravelled_free_positions.append(n)
        if self.scatter_free_agents:
            # We sort the available positions according to their distance from target
            # becuase we will grab the last position in the list when selecting
            # from the available cells, which will give us the furthest cells first.
            ravelled_free_positions.sort(
                key=lambda x: np.linalg.norm(
                    np.array([np.unravel_index(x, (self. rows, self.cols))]) - self._maze_start
                )
            )

        self.ravelled_positions_available = {
            **{
                encoding: deepcopy(ravelled_barrier_positions)
                for encoding in self.barrier_encodings
            },
            **{encoding: deepcopy(ravelled_free_positions) for encoding in self.free_encodings}
        }

    def _place_variable_position_agent(self, var_agent_to_place, **kwargs):
        """
        Place an agent with a variable position.

        This implementation places the agents according to their available positions,
        either a free cell or a barrier cell. Barriers agents will be clustered
        around the maze's starting position if cluster_barriers is True. Free
        agents will be scattered far from the maze's starting position if
        scatter_free_agents is True.
        """
        if (var_agent_to_place.encoding in self.barrier_encodings and self.cluster_barriers) \
            or (var_agent_to_place.encoding in self.free_encodings and
                self.scatter_free_agents):
            try:
                ravelled_position = \
                    self.ravelled_positions_available[var_agent_to_place.encoding][-1]
            except IndexError:
                raise RuntimeError(f"Could not find a cell for {var_agent_to_place.id}") from None
            else:
                r, c = np.unravel_index(ravelled_position, shape=(self.rows, self.cols))
                assert self.grid.place(var_agent_to_place, (r, c))
                self._update_available_positions(var_agent_to_place)
        else:
            super()._place_variable_position_agent(var_agent_to_place)


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


class AmmoState(StateBaseComponent):
    """
    Manage the state of the agents' ammo.

    Every AmmoAgent has ammo.
    """
    def reset(self, **kwargs):
        """
        Give AmmoAgents their starting ammo.
        """
        for agent in self.agents.values():
            if isinstance(agent, AmmoAgent):
                agent.ammo = agent.initial_ammo


class OrientationState(StateBaseComponent):
    """
    Manages the state of the agent's orientation.

    Orientation determines not only which way the agent is "facing" but also includes
    drift, which will move the agent one cell away in the direction that it is moving.
    """
    def reset(self, **kwargs):
        """
        Give OrientationAgents their initial orientation (or random if not assigned).
        """
        for agent in self.agents.values():
            if isinstance(agent, OrientationAgent):
                if agent.initial_orientation:
                    agent.orientation = agent.initial_orientation
                else:
                    agent.orientation = np.random.randint(1, 5)
