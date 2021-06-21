import numpy as np

from abmarl.sim import PrincipleAgent, ActingAgent, ObservingAgent


# ------------------ #
# --- Base Agent --- #
# ------------------ #


# --------------------- #
# --- Communication --- #
# --------------------- #



class BroadcastObservingAgent(ObservingAgent, ComponentAgent): pass


# ----------------------- #
# --- Health and Life --- #
# ----------------------- #

class LifeObservingAgent(ObservingAgent, ComponentAgent): pass
class HealthObservingAgent(ObservingAgent, ComponentAgent): pass


# ----------------------------- #
# --- Position and Movement --- #
# ----------------------------- #


class PositionObservingAgent(ObservingAgent, ComponentAgent): pass




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
