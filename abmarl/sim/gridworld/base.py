
from abc import ABC

import numpy as np

from abmarl.sim import AgentBasedSimulation
from abmarl.sim.gridworld.agent import GridWorldAgent
from abmarl.sim.gridworld.grid import Grid
from abmarl.tools.matplotlib_utils import mscatter


class GridWorldSimulation(AgentBasedSimulation, ABC):
    """
    GridWorldSimulation interface.

    Extends the AgentBasedSimulation interface for the GridWorld. We provide builders
    for streamlining the building process.

    Args:
        grid: The underlying grid. This is typically provided by the builder.
    """
    def __init__(self, grid=None, **kwargs):
        super().__init__(**kwargs)
        self.grid = grid

    @property
    def grid(self):
        """
        The underlying grid in the Grid World Simulation.
        """
        return self._grid

    @grid.setter
    def grid(self, value):
        assert isinstance(value, Grid), "Grid must be a Grid object."
        self._grid = value

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
    def build_sim_from_grid(cls, grid, extra_agents=None, **kwargs):
        """
        Build a GridSimluation from a Grid object.

        Args:
            grid: A Grid contains the all the agents index by location, so we can
                construct a simluation from it.
            extra_agents: A dictionary of agents which are not in the input grid
                but which we want to be a part of the simulation. Note: if there
                is an agent in the grid and in extra_agents, we will use the one
                from the grid.

        Returns:
            A GridSimulation built from the grid along with any extra agents.
        """
        assert type(grid) is Grid, "Grid object required."
        if extra_agents is not None:
            # We only check if it is a dictionary because that is the requirement
            # here. The ABM agents property will further check the contents of the
            # dictionary as needed.
            assert type(extra_agents) is dict, "Extra agents must be a dictionary."
            agents = extra_agents
        else:
            agents = {}
        for r in range(grid.rows):
            for c in range(grid.cols):
                if grid[r, c] is not None:
                    agents.update(grid[r, c])
                    for agent in grid[r, c].values():
                        np.testing.assert_array_equal(
                            agent.initial_position,
                            np.array([r, c])
                        ), "The initial position of the agent must match its position in the grid."

        return cls._build_sim(grid.rows, grid.cols, agents=agents, **kwargs)

    @classmethod
    def build_sim_from_array(cls, array, object_registry, extra_agents=None, **kwargs):
        """
        Build a GridSimulation from an array.

        Args:
            array: An array from which to build the initial grid. Each entry should
                be an alphanumeric character indicating which agent will be at that
                location. The agent will be given that initial position.
            object_registry: A dictionary that maps the characters in the array
                to a function that generates the agent with its unique id. Zeros,
                periods, and underscores are reserved for empty space.
            extra_agents: A dictionary of agents which are not in the input grid
                but which we want to be a part of the simulation. Note: if there
                is an agent in the array and in extra_agents, we will use the one
                from the array.

        Returns:
            A GridSimulation built from the array along with any extra agents.
        """
        assert type(array) is np.ndarray, "The array must be a numpy array."
        assert type(object_registry) is dict, "The object_registry must be a dictionary."
        assert all([i not in object_registry for i in [0, '.', '_']]), \
            "0, '.', and '_' are reserved for empty space."
        if extra_agents is not None:
            # We only check if it is a dictionary because that is the requirement
            # here. The ABM agents property will further check the contents of the
            # dictionary as needed.
            assert type(extra_agents) is dict, "Extra agents must be a dictionary."
            agents = extra_agents
        else:
            agents = {}
        n = 0
        rows = array.shape[0]
        cols = array.shape[1]
        for r in range(rows):
            for c in range(cols):
                char = array[r, c]
                if char in object_registry:
                    agent = object_registry[char](n)
                    agent.initial_position = np.array([r, c])
                    agents[agent.id] = agent
                    n += 1

        return cls._build_sim(rows, cols, agents=agents, **kwargs)

    @classmethod
    def build_sim_from_file(cls, file_name, object_registry, extra_agents=None, **kwargs):
        """
        Build a GridSimulation from a text file.

        Args:
            file_name: Name of the file that specifies the initial grid setup. In the file, each
                cell should be a single alphanumeric character indicating which agent
                will be at that position (from the perspective of looking down on the
                grid). That agent will be given that initial position.
            object_registry: A dictionary that maps characters from the file to a
                function that generates the agent. This must be a function because
                each agent must have unique id, which is generated here. Zeros,
                periods, and underscores are reserved for empty space.
            extra_agents: A dictionary of agents which are not in the input grid
                but which we want to be a part of the simulation. Note: if there
                is an agent in the file and in extra_agents, we will use the one
                from the file.

        Returns:
            A GridSimulation built from the file along with any extra agents.
        """
        assert type(file_name) is str, "The file_name must be the name of the file."
        assert type(object_registry) is dict, "The object_registry must be a dictionary."
        assert all([i not in object_registry for i in [0, '.', '_']]), \
            "0, '.', and '_' are reserved for empty space."
        if extra_agents is not None:
            # We only check if it is a dictionary because that is the requirement
            # here. The ABM agents property will further check the contents of the
            # dictionary as needed.
            assert type(extra_agents) is dict, "Extra agents must be a dictionary."
            agents = extra_agents
        else:
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
    def _build_sim(cls, rows, cols, **kwargs):
        grid = Grid(rows, cols, **kwargs)
        return cls(grid=grid, **kwargs)

    def render(self, fig=None, gridlines=True, background_color='w', **kwargs):
        """
        Draw the grid and all active agents in the grid.

        Agents are drawn at their positions using their respective shape and color.

        Args:
            fig: The figure on which to draw the grid. It's important
                to provide this figure because the same figure must be used when drawing
                each state of the simulation. Otherwise, a ton of figures will pop up,
                which is very annoying.
            gridlines: If true, then draw the gridlines.
            background_color: The background color of the grid, default is white.
        """
        draw_now = fig is None
        if draw_now:
            from matplotlib import pyplot as plt
            fig = plt.gcf()

        fig.clear()
        ax = fig.gca()
        ax.set_facecolor(background_color)

        # Draw the gridlines
        ax.set(xlim=(0, self.grid.cols), ylim=(0, self.grid.rows))
        ax.set_xticks(np.arange(0, self.grid.cols, 1))
        ax.set_yticks(np.arange(0, self.grid.rows, 1))
        if gridlines:
            ax.grid()

        # Draw the agents
        agents_x = [
            agent.position[1] + 0.5 for agent in self.agents.values() if agent.active
        ]
        agents_y = [
            self.grid.rows - 0.5 - agent.position[0]
            for agent in self.agents.values() if agent.active
        ]
        shape = [agent.render_shape for agent in self.agents.values() if agent.active]
        color = [agent.render_color for agent in self.agents.values() if agent.active]
        size = [agent.render_size for agent in self.agents.values() if agent.active]
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=size, facecolor=color)

        if draw_now:
            plt.plot()
            plt.pause(1e-17)


class GridWorldBaseComponent(ABC):
    """
    Component base class from which all components will inherit.

    Every component has access to the dictionary of agents and the grid.
    """
    def __init__(self, agents=None, grid=None, **kwargs):
        self.agents = agents
        self.grid = grid

    @property
    def rows(self):
        """
        The number of rows in the grid.
        """
        return self.grid.rows

    @property
    def cols(self):
        """
        The number of columns in the grid.
        """
        return self.grid.cols

    @property
    def grid(self):
        """
        The grid indexes the agents by their position.

        For example, an agent whose position is (3, 2) can be accessed through
        the grid with ``self.grid[3, 2]``. Components are responsible for maintaining
        the connection between agent position and grid index.
        """
        return self._grid

    @grid.setter
    def grid(self, value):
        assert isinstance(value, Grid), "The grid must be a Grid object."
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
