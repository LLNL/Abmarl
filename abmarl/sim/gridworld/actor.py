
from abc import ABC, abstractmethod

import numpy as np
from gym.spaces import Box, Discrete

from abmarl.sim.gridworld.base import GridWorldBaseComponent
from abmarl.sim.gridworld.agent import MovingAgent, AttackingAgent
import abmarl.sim.gridworld.utils as gu


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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.action_space[self.key] = Box(
                    -agent.move_range, agent.move_range, (2,), np.int
                )

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

        The agent's new position must be within the grid and the cell-occupation rules
        must be met.

        Args:
            agent: Move the agent if it is a MovingAgent.
            action_dict: The action dictionary for this agent in this step. If
                the agent is a MovingAgent, then the action dictionary will contain
                the "move" entry.

        Returns:
            True if the move is successful, False otherwise.
        """
        if isinstance(agent, self.supported_agent_type):
            action = action_dict[self.key]
            new_position = agent.position + action
            if 0 <= new_position[0] < self.rows and \
                    0 <= new_position[1] < self.cols:
                from_ndx = tuple(agent.position)
                to_ndx = tuple(new_position)
                if to_ndx == from_ndx:
                    return True
                elif self.grid.query(agent, to_ndx):
                    self.grid.remove(agent, from_ndx)
                    self.grid.place(agent, to_ndx)
                    return True
                else:
                    return False
            else:
                return False


class AttackActor(ActorBaseComponent):
    """
    Agents can attack other agents.
    """
    def __init__(self, attack_mapping=None, **kwargs):
        super().__init__(**kwargs)
        self.attack_mapping = attack_mapping
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.action_space[self.key] = Discrete(2)

    @property
    def attack_mapping(self):
        """
        Dict that dictates which agents the attacking agent can attack.

        The dictionary maps the attacking agents' encodings to a list of encodings
        that they can attack.
        """
        return self._attack_mapping

    @attack_mapping.setter
    def attack_mapping(self, value):
        assert type(value) is dict, "Attack mapping must be dictionary."
        for k, v in value.items():
            assert type(k) is int, "All keys in attack mapping must be integer."
            assert type(v) is list, "All values in attack mapping must be list."
            for i in v:
                assert type(i) is int, \
                    "All elements in the attack mapping values must be integers."
        self._attack_mapping = value

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

        1. The attacked agent is active.
        2. The attacked agent is within range.
        3. The attacked agent is valid according to the attack_mapping.

        If the attack is possible, then we determine the success of the attack
        based on the attacking agent's accuracy. If the attack is successful, then
        the attacked agent's health is depleted by the attacking agent's strength,
        possibly resulting in its death.
        """
        def determine_attack(agent):
            # Generate local grid and an attack mask.
            local_grid, mask = gu.create_grid_and_mask(
                agent, self.grid, agent.attack_range, self.agents
            )

            # Randomly scan the local grid for attackable agents.
            attackable_agents = []
            for r in range(2 * agent.attack_range + 1):
                for c in range(2 * agent.attack_range + 1):
                    if mask[r, c]: # We can see this cell
                        candidate_agents = local_grid[r, c]
                        if candidate_agents is not None:
                            for other in candidate_agents.values():
                                if other.id == agent.id: # Cannot attack yourself
                                    continue
                                elif not other.active: # Cannot attack inactive agents
                                    continue
                                elif other.encoding not in self.attack_mapping[agent.encoding]:
                                    # Cannot attack this type of agent
                                    continue
                                elif np.random.uniform() > agent.attack_accuracy:
                                    # Failed attack
                                    continue
                                else:
                                    attackable_agents.append(other)
            return np.random.choice(attackable_agents) if attackable_agents else None

        if isinstance(attacking_agent, self.supported_agent_type):
            action = action_dict[self.key]
            if action: # Agent has chosen to attack
                attacked_agent = determine_attack(attacking_agent)
                if attacked_agent is not None:
                    attacked_agent.health = attacked_agent.health - attacking_agent.attack_strength
                    if not attacked_agent.active:
                        self.grid.remove(attacked_agent, attacked_agent.position)
                return attacked_agent
