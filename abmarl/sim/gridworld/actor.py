
from abc import ABC, abstractmethod

import numpy as np
from gym.spaces import Box, Discrete

from abmarl.sim.gridworld.base import GridWorldBaseComponent
from abmarl.sim.gridworld.state import HealthState, PositionState
from abmarl.sim.gridworld.agent import MovingAgent, AttackingAgent, HealthAgent


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
    def __init__(self, position_state=None, **kwargs):
        super().__init__(**kwargs)
        self.position_state = position_state
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.action_space[self.key] = Box(
                    -agent.move_range, agent.move_range, (2,), np.int
                )

    @property
    def position_state(self):
        """
        PositionState component that manages the state of the agents' positions.
        """
        return self._position_state

    @position_state.setter
    def position_state(self, value):
        assert isinstance(value, PositionState), "Position state must be a PositionState object."
        self._position_state = value

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
        if isinstance(agent, self.supported_agent_type):
            action = action_dict[self.key]
            new_position = agent.position + action
            if 0 <= new_position[0] < self.rows and \
                    0 <= new_position[1] < self.cols and \
                    self.grid[new_position[0], new_position[1]] is None:
                self.grid[agent.position[0], agent.position[1]] = None
                agent.position = new_position
                self.grid[agent.position[0], agent.position[1]] = agent


class AttackActor(ActorBaseComponent):
    """
    Agents can attack other agents.

    The other agent must be within the attack_range of the attacking agent. If
    there are multiple attackable agents in the range, then one will be randomly
    chosen. The effectiveness of the attack is determined by the attacking agent's
    strength and accuracy.
    """
    def __init__(self, health_state=None, **kwargs):
        super().__init__(**kwargs)
        self.health_state = health_state
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.action_space[self.key] = Discrete(2)

    @property
    def health_state(self):
        """
        HealthState component that manages the state of the agents' healths.
        """
        return self._health_state

    @health_state.setter
    def health_state(self, value):
        assert isinstance(value, HealthState), "Health state must be a HealthState object."
        self._health_state = value

    @property
    def key(self):
        """
        This Actor's key is "attack".
        """
        return 'attack'

    @property
    def supported_agent_type(self):
        """
        This Actor works with AttackingAgents.
        """
        return AttackingAgent

    def process_action(self, attacking_agent, action_dict, **kwargs):
        """
        If the agent has chosen to attack, then we process their attack.

        The processing goes through a series of checks. The attack is possible
        if there is an attacked agent such that:
        1. The attacked agent is a HealthAgent.
        2. The attacked agent is active.
        3. The attacked agent is within range.
        
        If the attack is possible, then we determine the success of the attack
        based on the attacking agent's accuracy. If the attack is successful, then
        the attacked agent's health is depleted by the attacking agent's strength,
        possibly resulting in its death.
        """
        def determine_attack(agent):
            # Generate a completely empty grid
            local_grid = np.empty(
                (agent.attack_range * 2 + 1, agent.attack_range * 2 + 1), dtype=object
            )

            # Copy the section of the grid around the agent's position
            (r, c) = agent.position
            r_lower = max([0, r - agent.attack_range])
            r_upper = min([self.rows - 1, r + agent.attack_range]) + 1
            c_lower = max([0, c - agent.attack_range])
            c_upper = min([self.cols - 1, c + agent.attack_range]) + 1
            local_grid[
                (r_lower+agent.attack_range-r):(r_upper+agent.attack_range-r),
                (c_lower+agent.attack_range-c):(c_upper+agent.attack_range-c)
            ] = self.grid[r_lower:r_upper, c_lower:c_upper]

            # Generate an observation mask. The agent's observation can be blocked
            # by other view-blocking agents, which hide the cells "behind" them. We
            # calculate the blocking by drawing rays from the center of the agent's
            # position to the edges of the other agents' cell. All cells that are "behind"
            # that cell and between the two rays are invisible to the observing agent.
            # In the mask, 1 means that the cell is visibile, 0 means that it is
            # invisible.
            mask = np.ones((2 * agent.attack_range + 1, 2 * agent.attack_range + 1))
            for other in self.agents.values():
                if other.view_blocking:
                    r_diff, c_diff = other.position - agent.position
                    # Ensure the other agent within the view range
                    if -agent.attack_range <= r_diff <= agent.attack_range and \
                            -agent.attack_range <= c_diff <= agent.attack_range:
                        if c_diff > 0 and r_diff == 0: # Other is to the right of agent
                            upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
                            lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
                            for c in range(c_diff, agent.attack_range+1):
                                for r in range(-agent.attack_range, agent.attack_range+1):
                                    if c == c_diff and r == r_diff: continue # don't mask the other
                                    if lower(c) < r < upper(c):
                                        mask[r + agent.attack_range, c + agent.attack_range] = 0
                        elif c_diff > 0 and r_diff > 0: # Other is below-right of agent
                            upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
                            lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
                            for c in range(c_diff, agent.attack_range+1):
                                for r in range(r_diff, agent.attack_range+1):
                                    if c == c_diff and r == r_diff: continue # Don't mask the other
                                    if lower(c) < r < upper(c):
                                        mask[r + agent.attack_range, c + agent.attack_range] = 0
                        elif c_diff == 0 and r_diff > 0: # Other is below the agent
                            left = lambda t: (c_diff - 0.5) / (r_diff - 0.5) * t
                            right = lambda t: (c_diff + 0.5) / (r_diff - 0.5) * t
                            for c in range(-agent.attack_range, agent.attack_range+1):
                                for r in range(r_diff, agent.attack_range+1):
                                    if c == c_diff and r == r_diff: continue # don't mask the other
                                    if left(r) < c < right(r):
                                        mask[r + agent.attack_range, c + agent.attack_range] = 0
                        elif c_diff < 0 and r_diff > 0: # Other is below-left of agent
                            upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
                            lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
                            for c in range(c_diff, -agent.attack_range-1, -1):
                                for r in range(r_diff, agent.attack_range+1):
                                    if c == c_diff and r == r_diff: continue # don't mask the other
                                    if lower(c) < r < upper(c):
                                        mask[r + agent.attack_range, c + agent.attack_range] = 0
                        elif c_diff < 0 and r_diff == 0: # Other is left of agent
                            upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
                            lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
                            for c in range(c_diff, -agent.attack_range-1, -1):
                                for r in range(-agent.attack_range, agent.attack_range+1):
                                    if c == c_diff and r == r_diff: continue # don't mask the other
                                    if lower(c) < r < upper(c):
                                        mask[r + agent.attack_range, c + agent.attack_range] = 0
                        elif c_diff < 0 and r_diff < 0: # Other is above-left of agent
                            upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
                            lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
                            for c in range(c_diff, -agent.attack_range - 1, -1):
                                for r in range(r_diff, -agent.attack_range - 1, -1):
                                    if c == c_diff and r == r_diff: continue # don't mask the other
                                    if lower(c) < r < upper(c):
                                        mask[r + agent.attack_range, c + agent.attack_range] = 0
                        elif c_diff == 0 and r_diff < 0: # Other is above the agent
                            left = lambda t: (c_diff - 0.5) / (r_diff + 0.5) * t
                            right = lambda t: (c_diff + 0.5) / (r_diff + 0.5) * t
                            for c in range(-agent.attack_range, agent.attack_range+1):
                                for r in range(r_diff, -agent.attack_range - 1, -1):
                                    if c == c_diff and r == r_diff: continue # don't mask the other
                                    if left(r) < c < right(r):
                                        mask[r + agent.attack_range, c + agent.attack_range] = 0
                        elif c_diff > 0 and r_diff < 0: # Other is above-right of agent
                            upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
                            lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
                            for c in range(c_diff, agent.attack_range+1):
                                for r in range(r_diff, -agent.attack_range - 1, -1):
                                    if c == c_diff and r == r_diff: continue # don't mask the other
                                    if lower(c) < r < upper(c):
                                        mask[r + agent.attack_range, c + agent.attack_range] = 0

            # Convolve the local grid with the mask.
            local_grid_size = (2 * agent.attack_range + 1)
            rs, cs = np.unravel_index(
                np.random.choice(local_grid_size ** 2, local_grid_size ** 2, False),
                shape=(local_grid_size, local_grid_size)
            )
            for r, c in zip(rs, cs):
                if mask[r, c]:
                    other = local_grid[r, c]
                    if other is None: # No agent here
                        continue
                    if other.id == agent.id: # Cannot attack yourself
                        continue
                    if not isinstance(other, HealthAgent): # Cannot attack this agent
                        continue
                    elif not other.active: # Cannot attack inactive agents
                        continue
                    elif np.random.uniform() > agent.attack_accuracy:
                        continue
                    else:
                        return other

        if isinstance(attacking_agent, self.supported_agent_type):
            action = action_dict[self.key]
            if action: # Agent has chosen to attack
                attacked_agent = determine_attack(attacking_agent)
                if attacked_agent is not None:
                    attacked_agent.health = attacked_agent.health - attacking_agent.attack_strength
                    if not attacked_agent.active:
                        self.grid[attacked_agent.position[0], attacked_agent.position[1]] = None
                        attacked_agent.position = None
