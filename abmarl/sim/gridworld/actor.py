
from abc import ABC, abstractmethod

import numpy as np
from gym.spaces import Box

from abmarl.sim.gridworld import GridWorldBaseComponent, MovingAgent
from abmarl.sim.gridworld.state import GridWorldState


class ActorBaseComponent(GridWorldBaseComponent, ABC):
    """
    Abstract Actor Component class from which all Actor Components will inherit.
    """
    @abstractmethod
    def process_action(self, agent, action_dict, **kwargs):
        """
        Process the agent's action.

        Args:
            agent: The acting agent.
            action_dict: The action dictionary for this agent in this step. The
                dictionary may have different entries, each of which will be processed
                by different Actors.
        """
        pass

    @property
    @abstractmethod
    def key(self):
        """
        The key in the action dictionary.

        The action space of all acting agents in the gridworld framework is a dict.
        We can build up complex action spaces with multiple components by
        assigning each component an entry in the action dictionary. Actions
        will be a dictionary even if your simulation only has one Actor.
        """
        pass

    @property
    @abstractmethod
    def supported_agent_type(self):
        """
        The type of Agent that this Actor works with.

        If an agent is this type, the Actor will add its entry to the
        agent's action space and will process actions for this agent.
        """
        pass


class MoveActor(ActorBaseComponent):
    """
    Agents can move to unoccupied nearby squares.
    """
    def __init__(self, grid_state=None, **kwargs):
        super().__init__(**kwargs)
        self.grid_state = grid_state
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.action_space[self.key] = Box(
                    -agent.move_range, agent.move_range, (2,), np.int
                )

    @property
    def grid_state(self):
        """
        GridWorldState object that tracks the state of the grid.
        """
        return self._grid_state

    @grid_state.setter
    def grid_state(self, value):
        assert isinstance(value, GridWorldState), "Grid state must be a GridState object."
        self._grid_state = value

    @property
    def key(self):
        """
        This Actor's key is "move".
        """
        return "move"

    @property
    def supported_agent_type(self):
        """
        This Actor works with MovingAgents.
        """
        return MovingAgent

    def process_action(self, agent, action_dict, **kwargs):
        """
        The agent can move to nearby squares.

        Args:
            agent: Move the agent if it is a MovingAgent.
            action_dict: The action dictionary for this agent in this step. If
                the agent is a MovingAgent, then the action dictionary will contain
                the "move" entry.
        """
        if isinstance(agent, MovingAgent):
            action = action_dict[self.key]
            self.grid_state.set_position(agent, agent.position + action)
