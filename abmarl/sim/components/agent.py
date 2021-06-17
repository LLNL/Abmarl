import numpy as np

from abmarl.sim import PrincipleAgent, ActingAgent, ObservingAgent


# ------------------ #
# --- Base Agent --- #
# ------------------ #

class ComponentAgent(PrincipleAgent):
    """
    Component Agents have a position, life, and team.

    initial_position (np.array or None):
        The desired starting position for this agent.

    min_health (float):
        The minimum health the agent can reach before it dies.
        Default 0.0.

    max_health (float):
        The maximum health the agent can reach.
        Default 1.0.

    initial_health (float or None):
        The initial health of the agent. The health will be set to this initial
        option at reset time.

    team (int or None):
        The agent's team. Teams are indexed starting from 1, with team 0 reserved
        for agents that are not on a team (None).
    """
    def __init__(self, initial_position=None, min_health=0.0, max_health=1.0, initial_health=None,
                 team=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_position = initial_position
        assert type(min_health) in [float, int] and type(max_health) in [float, int], \
            "Min and max health must be numerical."
        assert min_health <= max_health, \
            "The min health must be less than or equal to the max_health."
        self._min_max_health = np.array([min_health, max_health])
        self.initial_health = initial_health
        self.is_alive = True
        self.team = team

    @property
    def initial_position(self):
        return self._initial_position

    @initial_position.setter
    def initial_position(self, value):
        if value is not None:
            assert type(value) is np.ndarray, "Initial position must be a numpy array."
            assert value.shape == (2,), "Initial position must be a 2-element array."
            assert value.dtype in [np.int, np.float], "Initial position must be numerical."
        self._initial_position = value

    @property
    def min_health(self):
        return self._min_max_health[0]

    @property
    def max_health(self):
        return self._min_max_health[1]

    @property
    def initial_health(self):
        return self._initial_health

    @initial_health.setter
    def initial_health(self, value):
        if value is not None:
            assert type(value) in [float, int], "Initial health must be a float."
            assert self.min_health <= value <= self.max_health, \
                "Initial health must be between the min and max health."
        self._initial_health = value

    @property
    def team(self):
        return self._team

    @team.setter
    def team(self, value):
        if value is not None:
            assert type(value) is int, "Team must be an int."
            assert value != 0, "Team 0 is reserved for agents who do not have a team. " + \
                "Use a team number greater than 0."
            self._team = value
        else:
            self._team = 0

    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return super().configured and \
            self._min_max_health is not None and \
            self.is_alive is not None and \
            self.team is not None


# ----------------- #
# --- Attacking --- #
# ----------------- #

class AttackingAgent(ActingAgent, ComponentAgent):
    """
    Agents that can attack other agents.

    attack_range (int):
        The effective range of the attack. Can be used to determine if an attack
        is successful based on distance between agents.

    attack_strength (float):
        How effective the agent's attack is. This is applicable in situations where
        the agents' health is affected by attacks.

    attack_accuracy (float):
        The effective accuracy of the agent's attack. Should be between 0 and 1.
        To make deterministic attacks, use 1. Default is 1.
    """
    def __init__(self, attack_range=None, attack_strength=None, attack_accuracy=1, **kwargs):
        super().__init__(**kwargs)
        assert attack_range is not None, "attack_range must be a nonnegative integer"
        self.attack_range = attack_range
        assert attack_strength is not None, "attack_strength must be a nonnegative number"
        self.attack_strength = attack_strength
        self.attack_accuracy = attack_accuracy

    @property
    def configured(self):
        """
        The agent is successfully configured if the attack range and strength is
        specified.
        """
        return super().configured and self.attack_range is not None and \
            self.attack_strength is not None


# --------------------- #
# --- Communication --- #
# --------------------- #

class BroadcastingAgent(ActingAgent, ComponentAgent):
    """
    BroadcastingAgents can broadcast their observation within some range of their
    position.

    braodcast_range (int):
        The agent's broadcasting range.
    """
    def __init__(self, broadcast_range=None, **kwargs):
        super().__init__(**kwargs)
        self.broadcast_range = broadcast_range
        self.broadcasting = False

    @property
    def configured(self):
        """
        The agent is successfully configured if the broadcast range is specified.
        """
        return super().configured and self.broadcast_range is not None


class BroadcastObservingAgent(ObservingAgent, ComponentAgent): pass


# ----------------------- #
# --- Health and Life --- #
# ----------------------- #

class LifeObservingAgent(ObservingAgent, ComponentAgent): pass
class HealthObservingAgent(ObservingAgent, ComponentAgent): pass


# ----------------- #
# --- Observing --- #
# ----------------- #

# TODO: move this to a more specific location
class AgentObservingAgent(ObservingAgent, ComponentAgent):
    """
    Agents can observe other agents.

    agent_view (int):
        Any agent within this many spaces will be fully observed.
    """
    def __init__(self, agent_view=None, **kwargs):
        """
        Agents can see other agents up to some maximal distance away, indicated
        by the view.
        """
        super().__init__(**kwargs)
        assert agent_view is not None, "agent_view must be nonnegative integer"
        self.agent_view = agent_view

    @property
    def configured(self):
        """
        Agents are configured if the agent_view parameter is set.
        """
        return super().configured and self.agent_view is not None


# ----------------------------- #
# --- Position and Movement --- #
# ----------------------------- #


class PositionObservingAgent(ObservingAgent, ComponentAgent): pass


class GridMovementAgent(ActingAgent, ComponentAgent):
    """
    Agents can move up to some number of spaces away.

    move_range (int):
        The maximum number of cells away that the agent can move.
    """
    def __init__(self, move_range=None, **kwargs):
        super().__init__(**kwargs)
        assert move_range is not None, "move_range must be an integer"
        self.move_range = move_range

    @property
    def configured(self):
        """
        Agents are configured if the move_range parameter is set.
        """
        return super().configured and self.move_range is not None


class SpeedAngleAgent(ComponentAgent):
    """
    Agents have a speed and a banking angle which are used to determine how the
    agent moves around a continuous field.

    min_speed (float):
        The minimum speed this agent can travel.

    max_speed (float):
        The maximum speed this agent can travel.

    max_banking_angle (float):
        The maximum banking angle the agent can endure.

    initial_speed (float):
        The agent's initial speed.

    initial_banking_angle (float):
        The agent's initial banking angle.

    initial_ground_angle (float):
        The agent's initial ground angle.
    """
    def __init__(self, min_speed=0.25, max_speed=1.0, max_banking_angle=45, initial_speed=None,
                 initial_banking_angle=None, initial_ground_angle=None, **kwargs):
        super().__init__(**kwargs)
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.initial_speed = initial_speed
        self.speed = None # Should be set by the state handler

        self.max_banking_angle = max_banking_angle
        self.initial_banking_angle = initial_banking_angle
        self.initial_ground_angle = initial_ground_angle
        self.banking_angle = None # Should be set by the state handler

    @property
    def configured(self):
        return super().configured and self.min_speed is not None and self.max_speed is not None \
            and self.max_banking_angle is not None


class SpeedAngleActingAgent(ActingAgent, ComponentAgent):
    """
    Agents can change their speed and banking angles.

    max_acceleration (float):
        The maximum amount by which an agent can change its speed in a single time
        step.

    max_banking_angle_change (float):
        The maximum amount by which an agent can change its banking angle in a
        single time step.
    """
    def __init__(self, max_acceleration=0.25, max_banking_angle_change=30, **kwargs):
        super().__init__(**kwargs)
        self.max_acceleration = max_acceleration
        self.max_banking_angle_change = max_banking_angle_change

    @property
    def configured(self):
        return super().configured and self.max_acceleration is not None and \
            self.max_banking_angle_change is not None


class SpeedAngleObservingAgent(ObservingAgent, ComponentAgent): pass


class VelocityAgent(ComponentAgent):
    """
    Agents have a velocity which determines how it moves around in a continuous
    field. Agents can accelerate to modify their velocities.

    initial_velocity (np.array):
        Two-element float array that is the agent's initial velocity.

    max_speed (float):
        The maximum speed the agent can travel.

    max_acceleration (float):
        The maximum amount by which an agent can change its velocity in a single
        time step.
    """
    def __init__(self, initial_velocity=None, max_speed=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_velocity = initial_velocity
        self.max_speed = max_speed

    @property
    def configured(self):
        return super().configured and self.max_speed is not None


class AcceleratingAgent(ActingAgent, ComponentAgent):
    """
    Agents can accelerate to modify their velocities.

    max_acceleration (float):
        The maximum amount by which an agent can change its velocity in a single
        time step.
    """
    def __init__(self, max_acceleration=None, **kwargs):
        super().__init__(**kwargs)
        self.max_acceleration = max_acceleration

    @property
    def configured(self):
        return super().configured and self.max_acceleration is not None


class VelocityObservingAgent(ObservingAgent, ComponentAgent): pass


class CollisionAgent(PrincipleAgent):
    """
    Agents that have physical size and mass and can be used in collisions.

    size (float):
        The size of the agent.
        Default 1.

    mass (float):
        The mass of the agent.
        Default 1.
    """
    def __init__(self, size=1, mass=1, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.mass = mass

    @property
    def configured(self):
        return super().configured and self.size is not None and self.mass is not None


# -------------------------------- #
# --- Resources and Harvesting --- #
# -------------------------------- #

class HarvestingAgent(ActingAgent, ComponentAgent):
    """
    Agents can harvest resources.

    max_harvest (double):
        The maximum amount of resources the agent can harvest from the cell it
        occupies.
    """
    def __init__(self, max_harvest=None, **kwargs):
        super().__init__(**kwargs)
        assert max_harvest is not None, "max_harvest must be nonnegative number"
        self.max_harvest = max_harvest

    @property
    def configured(self):
        """
        Agents are configured if max_harvest is set.
        """
        return super().configured and self.max_harvest is not None


class ResourceObservingAgent(ObservingAgent, ComponentAgent):
    """
    Agents can observe the resources in the simulation.

    resource_view (int):
        Any resources within this range of the agent's position will be fully observed.
    """
    def __init__(self, resource_view=None, **kwargs):
        super().__init__(**kwargs)
        assert resource_view is not None, "resource_view must be nonnegative integer"
        self.resource_view = resource_view

    @property
    def configured(self):
        """
        Agents are configured if the resource_view parameter is set.
        """
        return super().configured and self.resource_view is not None


# ------------ #
# --- Team --- #
# ------------ #

class TeamObservingAgent(ObservingAgent, ComponentAgent): pass
