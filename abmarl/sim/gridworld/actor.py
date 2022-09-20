
from abc import ABC, abstractmethod

import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete, Dict

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
    Agents can move to nearby squares.
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

    Agents can choose to use up to some number of their attacks. For example,
    if an agent has an attack count of 3, then it can choose no attack, attack
    once, attack twice, or attack thrice. The BinaryAttackActor searches the
    nearby local grid defined by the agent's attack range for attackable agents,
    and randomly chooses from that set up to the number of attacks issued.
    """
    def __init__(self, attack_mapping=None, stacked_attacks=False, **kwargs):
        super().__init__(**kwargs)
        self.attack_mapping = attack_mapping
        self.stacked_attacks = stacked_attacks
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                agent.action_space[self.key] = Discrete(agent.attack_count + 1)
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
    def stacked_attacks(self):
        """
        Allows an agent to attack the same agent multiple times per step.

        When an agent has more than 1 attack per turn, this parameter allows
        them to use more than one attack on the same agent. Otherwise, the attacks
        will be applied to other agents, and if there are not enough attackable
        agents, then the extra attacks will be wasted.
        """
        return self._stacked_attacks

    @stacked_attacks.setter
    def stacked_attacks(self, value):
        assert type(value) is bool, "Stacked attacks must be a boolean."
        self._stacked_attacks = value

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

        The agent can attack up to the number of its attack count. Each attack is
        successful if there is an attackable agent such that:

        1. The attackable agent is active.
        2. The attackable agent is within range.
        3. The attackable agent is valid according to the attack_mapping.
        4. The attacking agent's accuracy is high enough.

        Furthemore, a single agent may only be attacked once if stacked_attacks
        is False. Additional attacks will be applied on other agents or wasted.

        If the attack is successful, then the attacked agent's health is depleted
        by the attacking agent's strength, possibly resulting in its death.
        """
        def determine_attack(agent, attack):
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
            if attackable_agents:
                if not self.stacked_attacks and attack > len(attackable_agents):
                    attack = len(attackable_agents)
                return np.random.choice(
                    attackable_agents, size=attack, replace=self.stacked_attacks
                )
            return []

        if isinstance(attacking_agent, self.supported_agent_type):
            action = action_dict[self.key]
            if action: # Agent has chosen to attack
                attacked_agents = determine_attack(attacking_agent, action)
                for attacked_agent in attacked_agents:
                    attacked_agent.health = attacked_agent.health - attacking_agent.attack_strength
                    if not attacked_agent.active:
                        self.grid.remove(attacked_agent, attacked_agent.position)
                return attacked_agents
        return []


class EncodingBasedAttackActor(ActorBaseComponent):
    """
    Agents can attack other agents.

    The attacking agent specifies how many attacks it would like to use per available
    encoding, based on its attack count and the attack mapping. For example, if
    the agent can attack encodings 1 and 2 and has up to 3 attacks available, then
    it may specify up to 3 attacks on encoding 1 and up to 3 attack on encoding 2.
    Agents with those encodings in the surrounding grid are liable to be attacked.
    """
    def __init__(self, attack_mapping=None, stacked_attacks=False, **kwargs):
        super().__init__(**kwargs)
        self.attack_mapping = attack_mapping
        self.stacked_attacks = stacked_attacks
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                attackable_encodings = self.attack_mapping[agent.encoding]
                agent.action_space[self.key] = Dict({
                    i: Discrete(agent.attack_count + 1) for i in attackable_encodings
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

    @property
    def stacked_attacks(self):
        """
        Allows an agent to attack the same agent multiple times per step.

        When an agent has more than 1 attack per encoding, this parameter allows
        them to use more than one attack on the same agent. Otherwise, the attacks
        will be applied to other agents, and if there are not enough attackable
        agents, then the extra attacks will be wasted.
        """
        return self._stacked_attacks

    @stacked_attacks.setter
    def stacked_attacks(self, value):
        assert type(value) is bool, "Stacked attacks must be a boolean."
        self._stacked_attacks = value

    def process_action(self, attacking_agent, action_dict, **kwargs):
        """
        Process the agent's attack.

        The agent indicates which encoding(s) to attack. Each attack is successful
        if there is an attackable agent such that:

        1. The attackable agent is active.
        2. The attackable agent is within range.
        3. The attackable agent is valid according to the attack_mapping.
        4. The attacking agent's accuracy is high enough.

        Furthemore, a single agent may only be attacked once if stacked_attacks
        is False. Additional attacks will be applied on other agents or wasted.

        If the attack is successful, then the attacked agent's health is depleted
        by the attacking agent's strength, possibly resulting in its death.
        """
        def determine_attack(agent, attack):
            # Generate local grid and an attack mask.
            local_grid, mask = gu.create_grid_and_mask(
                agent, self.grid, agent.attack_range, self.agents
            )

            # Randomly scan the local grid for attackable agents.
            attackable_agents = {encoding: [] for encoding in attack}
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
                                elif other.encoding not in self.attack_mapping[agent.encoding]:
                                    # Cannot attack this type of agent
                                    continue
                                elif np.random.uniform() > agent.attack_accuracy:
                                    # Failed attack
                                    continue
                                else:
                                    attackable_agents[other.encoding].append(other)
            attacked_agents = []
            for encoding, num_attacks in attack.items():
                if len(attackable_agents[encoding]) == 0:
                    continue
                elif not self.stacked_attacks and num_attacks > len(attackable_agents[encoding]):
                    num_attacks = len(attackable_agents[encoding])
                attacked_agents.extend(np.random.choice(
                    attackable_agents[encoding], size=num_attacks, replace=self.stacked_attacks
                ))
            return attacked_agents

        if isinstance(attacking_agent, self.supported_agent_type):
            action = action_dict[self.key]
            attacked_agents = determine_attack(attacking_agent, action)
            for attacked_agent in attacked_agents:
                if not attacked_agent.active: continue # Skip this agent since it is dead
                attacked_agent.health = attacked_agent.health - attacking_agent.attack_strength
                if not attacked_agent.active:
                    self.grid.remove(attacked_agent, attacked_agent.position)
            return attacked_agents
        return []


class RestrictedSelectiveAttackActor(ActorBaseComponent):
    """
    Agents can attack other agents around it.

    The attacking agent is given up to K attacks to use on a nearby grid.
    """
    def __init__(self, attack_mapping=None, **kwargs):
        super().__init__(**kwargs)
        self.attack_mapping = attack_mapping
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                grid_cells = (2 * agent.attack_range + 1) ** 2
                agent.action_space[self.key] = MultiDiscrete(
                    [grid_cells + 1] * agent.attack_count
                )
                agent.null_action[self.key] = np.zeros((agent.attack_count,), dtype=int)

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

        The agent indicates which cells in a surrounding grid to attack. It can
        attack up to K cells, depending on the number of attacks it has per turn.
        It can also choose not to use one of its attacks and choose not to attack
        at all by not using any of its attacks. For each attack, the processing
        goes through a series of checks. The attack is possible if there is an
        attacked agent such that:

        1. The attacked agent is active.
        2. The attacked agent is located at the attacked cell.
        3. The attacked agent is valid according to the attack_mapping.

        If the attack is possible, then we determine the success of the attack
        based on the attacking agent's accuracy. If the attack is successful, then
        the attacked agent's health is depleted by the attacking agent's strength,
        possibly resulting in its death.
        """
        # TODO: Can attacked agents be attacked more than once per turn? That is,
        # can the attacking agent use more than one attack on a single agent, compounding
        # its effect?
        def determine_attack(agent, attack):
            # Generate local grid and an attack mask.
            local_grid, mask = gu.create_grid_and_mask(
                agent, self.grid, agent.attack_range, self.agents
            )

            attacked_agents = []
            for raveled_cell in attack:
                if raveled_cell == 0:
                    # Agent has chosen not to use this attack
                    continue
                else:
                    raveled_cell -= 1
                    r = raveled_cell % (2 * agent.attack_range + 1)
                    c = int(raveled_cell / (2 * agent.attack_range + 1))
                    attackable_agents = []
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
                                elif other in attacked_agents:
                                    # Cannot attack this agent again.
                                    # TODO: Variation for attacking an agent more than once.
                                    continue
                                else:
                                    attackable_agents.append(other)
                    if attackable_agents:
                        attacked_agents.append(np.random.choice(attackable_agents))
            return attacked_agents

        if isinstance(attacking_agent, self.supported_agent_type):
            action = action_dict[self.key]
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
