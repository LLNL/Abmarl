
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


class AttackActorBaseComponent(ActorBaseComponent, ABC):
    """
    Abstract class that provides the properties and structure for attack actors.

    The agent chooses to attack other agents within its surrounding grid. The derived
    attack actor interprets and implements the specific attack. Attacked agents
    have their health reduced by the attacking agent's strength and possibly become
    inactive if their health falls too low.
    """
    def __init__(self, attack_mapping=None, stacked_attacks=False, **kwargs):
        super().__init__(**kwargs)
        self.attack_mapping = attack_mapping
        self.stacked_attacks = stacked_attacks
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                self._assign_space(agent)

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

    def process_action(self, attacking_agent, action_dict, **kwargs):
        """
        Process the agent's attack.

        The derived attack actor interprets and implements the action. In general,
        an attack is successful if there are attackable agents such that:

        1. The attackable agent is active.
        2. The attackable agent is positioned at the attacked cell.
        3. The attackable agent is valid according to the attack_mapping.
        4. The attacking agent's accuracy is high enough.

        Furthemore, a single agent may only be attacked once if stacked_attacks
        is False. Additional attacks will be applied on other agents or wasted.

        If the attack is successful, then the attacked agent's health is depleted
        by the attacking agent's strength, possibly resulting in its death.

        Args:
            attacking_agent: The attacking agent.
            action_dict: The agent's action in this step.

        Returns:
            Tuple of (bool, list). The first value is False if the agent is not an
            attacking agent or chose not to attack; otherwise it is True. The second
            value is a list of attacked agents, which will be empty if there was
            no attack or if the attack failed. Thus, there are three possible outcomes:

            1. An attack was not attempted: False, []
            2. An attack failed: True, []
            3. An attack was successful: True, [non-empty]
        """
        if isinstance(attacking_agent, self.supported_agent_type):
            action = action_dict[self.key]
            attack_status, attacked_agents = self._determine_attack(attacking_agent, action)
            for attacked_agent in attacked_agents:
                if not attacked_agent.active: continue # Skip this agent since it is dead
                attacked_agent.health = attacked_agent.health - attacking_agent.attack_strength
                if not attacked_agent.active:
                    self.grid.remove(attacked_agent, attacked_agent.position)
            return attack_status, attacked_agents
        else:
            return False, []

    def _basic_criteria(self, attacking_agent, candidate):
        """
        Basic criteria shared among attack actors for attack success.

        The basic criteria shared amoung all attacking actors is the following:
        1. The candidate must not be the same as the attacking agent.
        2. The candidate must be active.
        3. The candidate's encoding must be in the attacker's attack mapping.
        4. The attacker's accuracy must be higher than a random number.

        If each of these criteria's is met, then the attack is succesful.

        Args:
            attacking_agent: The attacking agent.
            candidate: The candidate agent.
        Returns:
            True if the attack is successful, otherwise False.
        """
        if candidate.id == attacking_agent.id: # Cannot attack yourself
            return False
        elif not candidate.active: # Cannot attack inactive agents
            return False
        elif candidate.encoding not in self.attack_mapping[attacking_agent.encoding]:
            # Cannot attack this type of agent
            return False
        elif np.random.uniform() > attacking_agent.attack_accuracy:
            # Failed attack
            return False
        else:
            return True

    def _subset_attackables(self, attackable_agents, number_of_attacks):
        """
        Subset the list of attackable agents by the number of attacks.

        If the number of attacks is greater than the list of agents and stacked
        attacks is False, then each attackable agent is attacked, and any extra
        attacks are wasted. Otherwise,, randomly choose agents from the list of
        attackable_agents, using replacement if stacked attacks is True.

        Args:
            attackable_agents: List of attackable agents.
            number_of_attacks: The number agents to choose from the list. If stacked
                attacks is True, then the same agent could be chosen multiple times.
        Returns:
            List of attacked agents chosen from the list of attackable agents.
        """
        if not self.stacked_attacks and number_of_attacks > len(attackable_agents):
            return attackable_agents
        return np.random.choice(
            attackable_agents, size=number_of_attacks, replace=self.stacked_attacks
        )

    @abstractmethod
    def _determine_attack(self, agent, attack):
        """
        The derived class should determine which agents are successfully attacked.

        This function should return two values: a boolean indicating if an attack
        was attempted and a list of attacked agents. If the agent is not an attacking
        agent or chose not to attack, then the first value will be False; otherwise
        it will be True. The second value is a list of attacked agents, which will be
        empty if there was no attack or if the attack failed. Thus, there are
        three possible outcomes:
            1. An attack was not attempted: False, []
            2. An attack failed: True, []
            3. An attack was successful: True, [non-empty]
        """
        pass

    @abstractmethod
    def _assign_space(self, agent):
        """
        The derived class should assign the agent's action space and null_action.
        """
        pass


class BinaryAttackActor(AttackActorBaseComponent):
    """
    Launch attacks in a local grid.

    Agents can choose to launch attacks up to their `attack count` or not to attack
    at all. For example, if an agent has an attack count of 3, then it can choose
    no attack, attack once, attack twice, or attack thrice. The BinaryAttackActor
    searches the nearby local grid defined by the agent's attack range for attackable
    agents, and randomly chooses from that set up to the number of attacks issued.
    """
    def _assign_space(self, agent):
        agent.action_space[self.key] = Discrete(agent.attack_count + 1)
        agent.null_action[self.key] = 0

    def _determine_attack(self, agent, attack):
        """
        Process the agent's attack.

        The agent specifies how many attacks to carry out. The BinaryAttackActor
        searches the nearby local grid defined by the agent's attack range for attackable
        agents, and randomly chooses from that set up to the number of attacks issued.

        Args:
            agent: The attacking agent.
            attack: The number of attacks to perform.
        Returns:
            Tuple of bool, list. The first value is False if the agent is not an
            attacking agent or chose not to attack; otherwise it is True. The second
            value is a list of attacked agents, which will be empty if there was
            no attack or if the attack failed. Thus, there are three possible outcomes:
                1. An attack was not attempted: False, []
                2. An attack failed: True, []
                3. An attack was successful: True, [non-empty]
            The list of attacked agents will have length up to the number of attacks
            that the attacking agent can carry out per step. The agent's attack
            count is a total upper bound on the attack.
        """
        # Return empty list if no attack is specified.
        if not attack:
            return False, []

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
                            if self._basic_criteria(agent, other):
                                attackable_agents.append(other)

        if attackable_agents:
            return True, self._subset_attackables(attackable_agents, attack)
        else:
            return True, []


class EncodingBasedAttackActor(AttackActorBaseComponent):
    """
    Launch attacks in a local grid based on encoding.

    The attacking agent specifies how many attacks it would like to use per available
    encoding, based on its attack count and the attack mapping. For example, if
    the agent can attack encodings 1 and 2 and has up to 3 attacks available, then
    it may launch up to 3 attacks on encoding 1 and up to 3 attack on encoding 2.
    Agents with those encodings in the surrounding grid are liable to be attacked.
    """
    def _assign_space(self, agent):
        attackable_encodings = self.attack_mapping[agent.encoding]
        agent.action_space[self.key] = Dict({
            i: Discrete(agent.attack_count + 1) for i in attackable_encodings
        })
        agent.null_action[self.key] = {i: 0 for i in attackable_encodings}

    def _determine_attack(self, agent, attack):
        """
        Process the agent's attack.

        The agent specifies how many attacks to carry out per encoding. The
        EncodingBasedAttackActor searches the nearby local grid defined by the
        agent's attack range for attackable agents, and randomly chooses from that
        set up to the number of attacks issued for each encoding.

        Args:
            agent: The attacking agent.
            attack: The number of attacks to perform per each encoding.
        Returns:
            Tuple of bool, list. The first value is False if the agent is not an
            attacking agent or chose not to attack; otherwise it is True. The second
            value is a list of attacked agents, which will be empty if there was
            no attack or if the attack failed. Thus, there are three possible outcomes:
                1. An attack was not attempted: False, []
                2. An attack failed: True, []
                3. An attack was successful: True, [non-empty]
            The list of attacked agents will have length up to the number of attacks
            that the attacking agent can carry out per encoding per step. The agent's
            attack count is an upper bound on the attack per encoding, not a total
            upper bound.
        """
        # Return empty list if no attack is specified.
        if not any([num_attacks for num_attacks in attack.values()]):
            return False, []

        # Generate local grid and an attack mask.
        local_grid, mask = gu.create_grid_and_mask(
            agent, self.grid, agent.attack_range, self.agents
        )

        # Randomly scan the local grid for attackable agents.
        # This attack actor processes the attacking in the same way as the BinaryAttackActor;
        # the only difference is that it does the processing for each encoding.
        # We could have designed this differently to avoid code-duplication, but
        # we left it as is (1) for clarity of code and (2) because we believe our
        # implementation is more performant since we only have to scan the local
        # grid a single time.
        attackable_agents = {encoding: [] for encoding in attack}
        for r in range(2 * agent.attack_range + 1):
            for c in range(2 * agent.attack_range + 1):
                if mask[r, c]: # We can see this cell
                    candidate_agents = local_grid[r, c]
                    if candidate_agents is not None:
                        for other in candidate_agents.values():
                            if self._basic_criteria(agent, other):
                                attackable_agents[other.encoding].append(other)

        attacked_agents = []
        for encoding, num_attacks in attack.items():
            if len(attackable_agents[encoding]) == 0:
                continue
            attacked_agents.extend(
                self._subset_attackables(attackable_agents[encoding], num_attacks)
            )

        return True, attacked_agents


class RestrictedSelectiveAttackActor(AttackActorBaseComponent):
    """
    Launch attacks in a local grid by cell.

    Agents choose to attack specific cells in the surrounding grid. The agent can
    attack up to its attack count. It can choose to attack different cells or the
    same cell multiple times.
    """
    def _assign_space(self, agent):
        grid_cells = (2 * agent.attack_range + 1) ** 2
        agent.action_space[self.key] = MultiDiscrete(
            [grid_cells + 1] * agent.attack_count
        )
        agent.null_action[self.key] = np.zeros((agent.attack_count,), dtype=int)

    def _determine_attack(self, agent, attack):
        """
        Process the agent's attack.

        The agent specifies which grid cells to attack. The RestrictedSelectiveAttackActor
        randomly chooses attackable agents on each attacked cell.

        Args:
            agent: The attacking agent.
            attack: The nearby cells to attack.
        Returns:
            Tuple of bool, list. The first value is False if the agent is not an
            attacking agent or chose not to attack; otherwise it is True. The second
            value is a list of attacked agents, which will be empty if there was
            no attack or if the attack failed. Thus, there are three possible outcomes:
                1. An attack was not attempted: False, []
                2. An attack failed: True, []
                3. An attack was successful: True, [non-empty]
            The list of attacked agents will have length up to the number of attacks
            that the attacking agent can carry out per step. The agent's attack
            count is a total upper bound on the attack.
        """
        # Return empty list if no attack is specified.
        if not any(attack):
            return False, []

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
                # Agent has chosen to attack. We remove the "no attack" option
                # and then unravel the number to get the cell that is attacked.
                raveled_cell -= 1
                r = raveled_cell % (2 * agent.attack_range + 1)
                c = int(raveled_cell / (2 * agent.attack_range + 1))
                attackable_agents = []
                if mask[r, c]: # We can see this cell
                    candidate_agents = local_grid[r, c]
                    if candidate_agents is not None:
                        for other in candidate_agents.values():
                            if not self._basic_criteria(agent, other):
                                continue
                            elif other in attacked_agents and not self.stacked_attacks:
                                # Cannot attack this agent again.
                                continue
                            else:
                                attackable_agents.append(other)

                if attackable_agents:
                    attacked_agents.append(np.random.choice(attackable_agents))

        return True, attacked_agents


class SelectiveAttackActor(AttackActorBaseComponent):
    """
    Launch attacks in a local grid by cell.

    The attack is a local grid centered on the agent's position, and its size depends
    on the agent's attack range. Each cell in the grid has a nonnegative integer
    up to the agent's attack count, and it indicates how many attacks to use on
    that cell.
    """
    def _assign_space(self, agent):
        agent.action_space[self.key] = Box(
            0, agent.attack_count, (2 * agent.attack_range + 1, 2 * agent.attack_range + 1), int
        )
        agent.null_action[self.key] = np.zeros(
            (2 * agent.attack_range + 1, 2 * agent.attack_range + 1), dtype=int
        )

    def _determine_attack(self, agent, attack):
        """
        Process the agent's attack.

        The agent specifies which grid cells to attack. The SelectiveAttackActor
        randomly chooses attackable agents on each attacked cell.

        Args:
            agent: The attacking agent.
            attack: The nearby cells to attack.
        Returns:
            Tuple of bool, list. The first value is False if the agent is not an
            attacking agent or chose not to attack; otherwise it is True. The second
            value is a list of attacked agents, which will be empty if there was
            no attack or if the attack failed. Thus, there are three possible outcomes:
                1. An attack was not attempted: False, []
                2. An attack failed: True, []
                3. An attack was successful: True, [non-empty]
            The list of attacked agents will have length up to the number of attacks
            that the attacking agent can carry out per cell per step. The agent's
            attack count is an upper bound on the attack per cell, not a total
            upper bound.
        """
        # Return empty list if no attack is specified.
        if not np.any(attack):
            return False, []

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
                    candidate_agents = local_grid[r, c]
                    if candidate_agents is not None:
                        for other in candidate_agents.values():
                            if self._basic_criteria(agent, other):
                                attackable_agents.append(other)

                if attackable_agents:
                    attacked_agents.extend(
                        self._subset_attackables(attackable_agents, attack[r, c])
                    )

        return True, attacked_agents
