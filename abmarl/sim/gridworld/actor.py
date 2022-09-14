
from abc import ABC, abstractmethod

import numpy as np
from gym.spaces import Box, Discrete, Dict

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
                    -agent.move_range, agent.move_range, (2,), int
                )
                agent.null_action[self.key] = np.zeros((2,), dtype=int)

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


class BinaryAttackActor(ActorBaseComponent):
    """
    Agents can attack other agents.
    """
    def __init__(self, attack_mapping=None, **kwargs):
        super().__init__(**kwargs)
        self.attack_mapping = attack_mapping
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.action_space[self.key] = Discrete(2)
                agent.null_action[self.key] = 0

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


class EncodingBasedAttackActor(ActorBaseComponent):
    """
    Agents can attack other agents around it.

    The attacking agent specifies which encoding it would like to attack. If that
    encoding is nearby, then the attack may be succesful. Note: multiple encodings
    can be attacked at the same time.
    """
    def __init__(self, attack_mapping=None, **kwargs):
        super().__init__(**kwargs)
        self.attack_mapping = attack_mapping
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                attackable_encodings = self.attack_mapping[agent.encoding]
                agent.action_space[self.key] = Dict({
                    i: Discrete(2) for i in attackable_encodings
                })
                agent.null_action[self.key] = {i: 0 for i in attackable_encodings}

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
        Process the agent's attack.

        The agent indicates which encoding(s) to attack. For each encoding

        If the agent has chosen to attack, then we process their attack. The attack
        is possible if there is an attacked agent such that:

        1. The attacked agent is active.
        2. The attacked agent is within range.
        3. The attacked agent has attacked encoding.

        If the attack is possible, then we determine the success of the attack
        based on the attacking agent's accuracy. If the attack is successful, then
        the attacked agent's health is depleted by the attacking agent's strength,
        possibly resulting in its death.

        Only one agent per encoding can be attacked. If there are multiple agents
        with the same encoding, then we randomly pick one of them.
        """
        def determine_attack(agent, attack):
            # Generate local grid and an attack mask.
            local_grid, mask = gu.create_grid_and_mask(
                agent, self.grid, agent.attack_range, self.agents
            )

            # Randomly scan the local grid for attackable agents.
            attackable_agents = {encoding: [] for encoding, activated in attack.items() if activated}
            for r in range(2 * agent.attack_range + 1):
                for c in range(2 * agent.attack_range + 1):
                    if mask[r, c]: # We can see this cell
                        # TODO: Variation for masked cell?
                        candidate_agents = local_grid[r, c]
                        if candidate_agents is not None:
                            for other in candidate_agents.values():
                                if other.id == agent.id: # Cannot attack yourself
                                    continue
                                elif not other.active: # Cannot attack inactive agents
                                    continue
                                elif other.encoding not in attackable_agents:
                                    # Did not attack this encoding
                                    continue
                                elif np.random.uniform() > agent.attack_accuracy:
                                    # Failed attack
                                    continue
                                else:
                                    attackable_agents[other.encoding].append(other)
            attacked_agents = [
                np.random.choice(agent_list)
                for agent_list in attackable_agents.values() if agent_list
            ]
            # TODO: Variation for attacking all agents here?
            return attacked_agents

        if isinstance(attacking_agent, self.supported_agent_type):
            action = action_dict[self.key]
            # if np.any(action): # Agent has chosen to attack
            attacked_agents = determine_attack(attacking_agent, action)
            for attacked_agent in attacked_agents:
                attacked_agent.health = attacked_agent.health - attacking_agent.attack_strength
                if not attacked_agent.active:
                    self.grid.remove(attacked_agent, attacked_agent.position)
            return attacked_agents
        else:
            return []


class SelectiveAttackActor(ActorBaseComponent):
    """
    Agents can attack other agents around it.

    The attack is a grid of 1s and 0s, 1 being attack and 0 being don't attack.
    The grid is centered on the agent's location.
    """
    def __init__(self, attack_mapping=None, **kwargs):
        super().__init__(**kwargs)
        self.attack_mapping = attack_mapping
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.action_space[self.key] = Box(
                    0, 1, (2 * agent.attack_range + 1, 2 * agent.attack_range + 1), int
                )
                agent.null_action[self.key] = np.zeros(
                    (2 * agent.attack_range + 1, 2 * agent.attack_range + 1), dtype=int
                )

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
        return "attack"

    @property
    def supported_agent_type(self):
        """
        This Actor works with AttackingAgents.
        """
        return AttackingAgent

    def process_action(self, attacking_agent, action_dict, **kwargs):
        """
        Process the agent's attack.

        The agent indicates which cells in a surrounding grid to attack. If the
        cell value is 1, then it attacks, otherwise it does not attack. For each
        attack, the processing goes through a series of checks. The attack is possible
        if there is an attacked agent such that:

        1. The attacked agent is active.
        2. The attacked agent is within range.
        3. The attacked agent is valid according to the attack_mapping.

        If the attack is possible, then we determine the success of the attack
        based on the attacking agent's accuracy. If the attack is successful, then
        the attacked agent's health is depleted by the attacking agent's strength,
        possibly resulting in its death.
        """
        def determine_attack(agent, attack):
            # Generate local grid and an attack mask.
            local_grid, mask = gu.create_grid_and_mask(
                agent, self.grid, agent.attack_range, self.agents
            )

            attacked_agents = []
            for r in range(2 * agent.attack_range + 1):
                for c in range(2 * agent.attack_range + 1):
                    attackable_agents = []
                    if not attack[r, c]: continue # Agent did not attack here
                    if mask[r, c]: # We can see this cell
                        # TODO: Variation for masked cell?
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
                    if attackable_agents:
                        attacked_agents.append(np.random.choice(attackable_agents))
                        # TODO: Variation for attacking all agents here?
            return attacked_agents

        if isinstance(attacking_agent, self.supported_agent_type):
            action = action_dict[self.key]
            # if np.any(action): # Agent has chosen to attack
            attacked_agents = determine_attack(attacking_agent, action)
            for attacked_agent in attacked_agents:
                attacked_agent.health = attacked_agent.health - attacking_agent.attack_strength
                if not attacked_agent.active:
                    self.grid.remove(attacked_agent, attacked_agent.position)
            return attacked_agents
        else:
            return []
