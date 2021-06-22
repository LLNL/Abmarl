
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
            self.position_state.update(agent, new_position)


class AttackActor(ActorBaseComponent):
    """
    Agents can attack other agents.

    The other agent must be within the attack_range of the attacking agent. If
    there are multiple attackable agents in the range, then one will be randomly
    chosen. The effectiveness of the attack is determined by the attacking agent's
    strength and accuracy.
    """
    def __init__(self, health_state=None, attack_norm=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.health_state = health_state
        self.attack_norm = attack_norm
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
    def attack_norm(self):
        """
        Norm used to measure the distance between agents.

        Attack norm must be a positive integer or np.inf.
        """
        return self._attack_norm

    @attack_norm.setter
    def attack_norm(self, value):
        assert (type(value) is int and 0 < value) or value == np.inf, \
            "Attack norm must be a positive integer or np.inf"
        self._attack_norm = value

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
        the attacking agent's strength is given to the Health State, which manages
        the effects of the attack and updates the simulation state.
        """
        # "Kernel" for determining if an agent was attacked
        # TODO: search the nearby grid, not the dict of agents.
        # TODO: Walls should block attack like they block view.
        def determine_attack(attacking_agent):
            for attacked_agent in self.agents.values():
                if attacked_agent.id == attacking_agent.id:
                    # Cannot attack yourself
                    continue
                elif not isinstance(attacked_agent, HealthAgent):
                    # Attacked agent must be "attackable"
                    continue
                elif not attacked_agent.active:
                    # Cannot attack a dead agent
                    continue
                elif np.linalg.norm(
                            attacking_agent.position - attacked_agent.position,
                            self.attack_norm
                        ) > attacking_agent.attack_range:
                    # Agent is too far away
                    continue
                elif np.random.uniform() > attacking_agent.attack_accuracy:
                    # Attempted attack, but it failed
                    continue
                else:
                    # The agent was successfully attacked!
                    return attacked_agent

        if isinstance(attacking_agent, self.supported_agent_type):
            action = action_dict[self.key]
            if action: # Agent has chosen to attack
                attacked_agent = determine_attack(attacking_agent)
                if attacked_agent is not None:
                    self.health_state.update(
                        attacked_agent, attacked_agent.health - attacking_agent.attack_strength
                    )
