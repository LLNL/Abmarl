
from abc import ABC, abstractmethod

import numpy as np
from gym.spaces import Box, Discrete

from abmarl.sim.gridworld.base import GridWorldBaseComponent
from abmarl.sim.gridworld.state import HealthState
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
    chosen. Attackable agents are determiend by the team_matrix. The effectiveness
    of the attack is determined by the attacking agent's strength and accuracy.
    """
    def __init__(self, health_state=None, attack_norm=np.inf, team_attack_matrix=None,
            number_of_teams=None, **kwargs):
        super().__init__(**kwargs)
        self.health_state = health_state
        self.attack_norm = attack_norm
        self.number_of_teams = number_of_teams
        self.team_attack_matrix = team_attack_matrix
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
    def number_of_teams(self):
        """
        The number of teams in the simulation.

        The default is 0, which indicates that there are no teams, so attacking
        is "free-for-all".
        """
        return self._number_of_teams

    @number_of_teams.setter
    def number_of_teams(self, value):
        assert type(value) is int and 0 <= value, "Number of teams must be a nonnegative integer."
        self._number_of_teams = value

    @property
    def team_attack_matrix(self):
        """
        A matrix that indicates which teams can attack which other team.
        
        We use the value at the index, like so:
            team_matrix[attacking_team, attacked_team] = 0 or 1.
        0 indicates it cannot attack, 1 indicates that it can.

        If this is not specified, we build one that allows agents to attack any
        other team, including agents who do not have a team.
        """
        return self._team_attack_matrix

    @team_attack_matrix.setter
    def team_attack_matrix(self, value):
        if value is None:
            self._team_attack_matrix = -np.diag(np.ones(self.number_of_teams+1)) + 1
            self._team_attack_matrix[0, 0] = 1
        else:
            assert type(value) is np.ndarray, "Team attack matrix must be a numpy array."
            assert value.shape == (self.number_of_teams, self._number_of_teams), \
                "Team attack matrix size must be (number_of_teams x number_of_teams.)"
            self._team_attack_matrix = value

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
        2. The attacked agent is alive.
        3. The attacked agent is within range.
        4. The attacked agent is attackable according to the team attack matrix.
        
        If the attack is possible, then we determine the success of the attack
        based on the attacking agent's accuracy. If the attack is successful, then
        the attacking agent's strength is given to the Health State, which manages
        the effects of the attack and updates the simulation state.
        """
        # "Kernel" for determining if an agent was attacked
        def determine_attack(attacking_agent):
            for attacked_agent in self.agents.values():
                if attacked_agent.id == attacking_agent.id:
                    # Cannot attack yourself
                    continue
                elif not isinstance(attacked_agent, HealthAgent):
                    # Attacked agent must be "attackable"
                    continue
                elif not attacked_agent.is_alive:
                    # Cannot attack a dead agent
                    continue
                elif np.linalg.norm(
                            attacking_agent.position - attacked_agent.position,
                            self.attack_norm
                        ) > attacking_agent.attack_range:
                    # Agent is too far away
                    continue
                elif not self.team_attack_matrix[attacking_agent.team, attacked_agent.team]:
                    # Attacking agent cannot attack this agent
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
                    self.health_state.set_health(
                        attacked_agent, attacked_agent.health - attacking_agent.attack_strength
                    )
