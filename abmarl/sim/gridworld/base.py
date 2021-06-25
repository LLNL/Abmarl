
from abc import ABC, abstractmethod

import numpy as np

from abmarl.sim import AgentBasedSimulation
from abmarl.sim.gridworld.agent import GridWorldAgent


class Grid(np.ndarray, ABC):
    """
    A Grid is a numpy array that stores the positions of the agents.

    Manages the agent positions through the add and remove functions while abstracting
    the underlying datatype used to store the agent there.
    """
    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the grid to an empty state.
        """
        pass

    @abstractmethod
    def query(self, agent, ndx):
        """
        Query a cell in the grid to see if is available to this agent.

        Args:
            agent: The agent for which we are checking availabilty.
            ndx: The cell to query.

        Returns:
            True if the cell is available for this agent, otherwise False.
        """
        pass

    @abstractmethod
    def place(self, agent, ndx):
        """
        Place an agent at an index.

        If the placement is successful, the agent will be placed at that index
        in the grid and the agent's position will be updated.

        Args:
            agent: The agent to place.
            ndx: The new index for this agent.

        Returns:
            The successfulness of the placement.
        """
        pass

    @abstractmethod
    def move(self, agent, to_ndx):
        """
        Move an agent from one index to another.

        If the move is successful, the agent will be placed at that index in the
        grid and the agent's position will be updated.

        Note: the agent will be removed from its old position in the grid.

        Args:
            agent: The agent to move.
            to_ndx: The new index for this agent.

        Returns:
            The successfulness of the move.
        """
        pass

    @abstractmethod
    def remove(self, agent, ndx):
        """
        Remove an agent from an index.

        Args:
            agent: The agent to remove
            ndx: The old index for this agent.
        """
        pass

    @abstractmethod
    def _place(self, agent, ndx):
        """
        Unprotected placement. Internal use only.
        """
        pass


class NonOverlappingGrid(Grid):
    """
    A grid where agents cannot overlap.
    """
    def reset(self, **kwargs):
        self.fill(None)

    def query(self, agent, ndx):
        """
        The cell is available for the agent if it is empty.
        """
        ndx = tuple(ndx)
        return self[ndx] is None

    def place(self, agent, ndx):
        """
        The placement is succesful if the new position is unoccupied.
        """
        ndx = tuple(ndx)
        if self.query(agent, ndx):
            self._place(agent, ndx)
            agent.position = np.array(ndx)
            return True
        else:
            return False

    def move(self, agent, to_ndx):
        """
        The move is succesful if the new position is unoccupied.
        """
        from_ndx = tuple(agent.position)
        to_ndx = tuple(to_ndx)
        if self.query(agent, to_ndx):
            self._place(agent, to_ndx)
            self.remove(agent, from_ndx)
            agent.position = np.array(to_ndx)
            return True
        else:
            return False

    def remove(self, agent, ndx):
        ndx = tuple(ndx)
        self[ndx] = None

    def _place(self, agent, ndx):
        # Unprotected placement
        self[ndx] = agent


class OverlappableGrid(Grid):
    """
    A grid where agents can overlap.
    """
    def reset(self, **kwargs):
        self.fill(set())

    def query(self, agent, ndx):
        """
        The cell is available for the agent if it is empty or if both the occupying
        agent and the querying agent are overlappable.
        """
        ndx = tuple(ndx)
        return self[ndx] is None or (next(iter(self[ndx])).overlappable and agent.overlappable)

    def place(self, agent, ndx):
        """
        The placement is successful if the new position is unoccupied or if the
        agent already occupying that position is overlappable AND this agent is
        overlappable.
        """
        ndx = tuple(ndx)
        if self.query(agent, ndx):
            self._place(agent, ndx)
            agent.position = np.array(ndx)
            return True
        else:
            return False

    def move(self, agent, to_ndx):
        """
        The move is successful if the new position is unoccupied or if the agent
        already occupying that position is overlappable AND this agent is overlappable.
        """
        from_ndx = tuple(agent.position)
        to_ndx = tuple(to_ndx)
        if self.query(agent, to_ndx):
            self._place(agent, to_ndx)
            self.remove(agent, from_ndx)
            agent.position = np.array(to_ndx)
            return True
        else:
            return False

    def remove(self, agent, ndx):
        ndx = tuple(ndx)
        self[ndx].remove(agent)

    def _place(self, agent, ndx):
        # Unprotected placement
        self[ndx].add(agent)


class GridWorldSimulation(AgentBasedSimulation, ABC):
    """
    GridWorldSimulation interface.

    Extends the AgentBasedSimulation interface for the GridWorld. We provide builders
    for streamlining the building process. The interface is meant to be used like
    so:
    MyCustomGridSim(GridWorldSimulation):
        ... # define the simulation, don't overwrite the builders.

    sim = MyCustomGridSim.build(parameters)
    """
    @classmethod
    def build_sim(cls, rows, cols, **kwargs):
        """
        Build a GridSimulation.

        Specify the number of row, the number of cols, a dictionary of agents,
        and any additional parameters.

        Args:
            rows: The number of rows in the grid. Must be a positive integer.
            cols: The number of cols in the grid. Must be a positive integer.
            agents: The dictionary of agents in the grid.

        Returns:
            A GridSimulation configured as specified.
        """
        assert type(rows) is int, "Rows must be an integer."
        assert 0 < rows, "Rows must be a positive integer."
        assert type(cols) is int, "Cols must be an integer."
        assert 0 < cols, "Cols must be a positive integer."

        return cls._build_sim(rows, cols, **kwargs)

    @classmethod
    def build_sim_from_file(cls, file_name, object_registry, **kwargs):
        """
        Build a GridSimulation from a text file.

        Args:
            file_name: Name of the file that specifies the initial grid setup. In the file, each
                cell should be a single alphanumeric character indicating which agent
                will be at that position (from the perspective of looking down on the
                grid). That agent will be given that initial position. 0's are reserved for empty
                space. For example:
                0 A W W W A 0
                0 0 0 0 0 0 0
                0 A W W W A 0
                will create a 3-by-7 grid with some agents along the top and bottom
                of the grid and another type of agent in the corners.
            object_registry: A dictionary that maps characters from the file to a
                function that generates the agent. This must be a function because
                each agent must have unique id, which is generated here. For example,
                using the grid above and some pre-created Agent classes:
                {
                    'A': lambda n: ExploringAgent(id=f'explorer{n}', view_range=3, move_range=1),
                    'W': lambda n: WallAgent(id=f'wall{n}')
                }
                will create a grid with ExploringAgents in the corners and WallAgents
                along the top and bottom rows.

        Returns:
            A GridSimulation built from the file.
        """
        assert type(file_name) is str, "The file_name must be the name of the file."
        assert type(object_registry) is dict, "The object_registry must be a dictionary."
        assert 0 not in object_registry, "0 is reserved for empty space."
        agents = {}
        n = 0
        with open(file_name, 'r') as fp:
            lines = fp.read().splitlines()
            cols = len(lines[0].split(' '))
            rows = len(lines)
            for row, line in enumerate(lines):
                chars = line.split(' ')
                assert len(chars) == cols, f"Mismatched number of columns per row in {file_name}"
                for col, char in enumerate(chars):
                    if char in object_registry:
                        agent = object_registry[char](n)
                        agent.initial_position = np.array([row, col])
                        agents[agent.id] = agent
                        n += 1

        return cls._build_sim(rows, cols, agents=agents, **kwargs)

    @classmethod
    def _build_sim(cls, rows, cols, overlapping=False, **kwargs):
        if overlapping:
            grid = OverlappableGrid((rows, cols), dtype=object)
        else:
            grid = NonOverlappingGrid((rows, cols), dtype=object)
        kwargs['grid'] = grid
        return cls(**kwargs)


class GridWorldBaseComponent(ABC):
    """
    Component base class from which all components will inherit.

    Every component has access to the dictionary of agents and the grid.
    """
    def __init__(self, agents=None, grid=None, **kwargs):
        self.agents = agents
        self.grid = grid
        self._rows = self.grid.shape[0]
        self._cols = self.grid.shape[1]

    @property
    def rows(self):
        """
        The number of rows in the grid.
        """
        return self._rows

    @property
    def cols(self):
        """
        The number of columns in the grid.
        """
        return self._cols

    @property
    def grid(self):
        """
        The grid indexes the agents by their position.

        For example, an agent whose position is (3, 2) can be accessed through
        the grid with self.grid[3, 2]. Components are responsible for maintaining
        the connection between agent position and grid index.
        """
        return self._grid

    @grid.setter
    def grid(self, value):
        assert isinstance(value, Grid), "The grid must be a Grid object."
        assert len(value.shape) == 2, "The grid must be a 2-dimensional array."
        assert value.dtype is np.dtype(object), "The grid must be a numpy array of objects."
        self._grid = value

    @property
    def agents(self):
        """
        A dict that maps the Agent's id to the Agent object. All agents must be
        GridWorldAgents.
        """
        return self._agents

    @agents.setter
    def agents(self, value_agents):
        assert type(value_agents) is dict, "Agents must be a dict."
        for agent_id, agent in value_agents.items():
            assert isinstance(agent, GridWorldAgent), \
                "Values of agents dict must be instance of GridWorldAgent."
            assert agent_id == agent.id, \
                "Keys of agents dict must be the same as the Agent's id."
        self._agents = value_agents
