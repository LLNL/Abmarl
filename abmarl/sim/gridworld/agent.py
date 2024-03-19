
import numpy as np

from abmarl.sim import PrincipleAgent, ActingAgent, ObservingAgent


class GridWorldAgent(PrincipleAgent):
    """
    The base agent in the GridWorld.
    """
    def __init__(self, initial_position=None, blocking=False, encoding=None, render_shape='o',
                 render_color='gray', render_size=200, **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.initial_position = initial_position
        self.blocking = blocking
        self.render_shape = render_shape
        self.render_color = render_color
        self.render_size = render_size

    @property
    def encoding(self):
        """
        The numerical value that identifies the type of agent.

        The value does not necessarily identify the agent itself. For example,
        other agents who observe this agent will see this value.
        """
        return self._encoding

    @encoding.setter
    def encoding(self, value):
        assert type(value) is int, f"{self.id}'s encoding must be an integer."
        assert value != -2, "-2 encoding reserved for masked observation."
        assert value != -1, "-1 encoding reserved for out of bounds."
        assert value != 0, "0 encoding reserved for empty cell."
        self._encoding = value

    @property
    def initial_position(self):
        """
        The agent's initial position at reset.
        """
        return self._initial_position

    @initial_position.setter
    def initial_position(self, value):
        if value is not None:
            assert type(value) is np.ndarray, "Initial position must be a numpy array."
            assert value.shape == (2,), "Initial position must be a 2-element array."
            assert value.dtype in [int, float], "Initial position must be numerical."
        self._initial_position = value

    @property
    def position(self):
        """
        The agent's position in the grid.
        """
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def blocking(self):
        """
        Specify if this agent blocks other agent's observations and actions.
        """
        return self._blocking

    @blocking.setter
    def blocking(self, value):
        assert type(value) is bool, "Blocking must be either True or False."
        self._blocking = value

    @property
    def render_shape(self):
        """
        The agent's shape in the rendered grid.
        """
        return self._render_shape

    @render_shape.setter
    def render_shape(self, value):
        assert value in [
            'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p',
            'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd'
        ], "Invalid render shape."
        self._render_shape = value

    @property
    def render_color(self):
        """
        The agent's color in the rendered grid.
        """
        return self._render_color

    @render_color.setter
    def render_color(self, value):
        self._render_color = value

    @property
    def render_size(self):
        """
        The agent's size in the rendered grid.
        """
        return self._render_size

    @render_size.setter
    def render_size(self, value):
        assert type(value) is int and value > 0, "Render size must be nonnegative integer."
        self._render_size = value

    @property
    def configured(self):
        return super().configured and self.encoding is not None and \
            self.blocking is not None and self.render_shape is not None and \
            self.render_size is not None


class GridObservingAgent(ObservingAgent, GridWorldAgent):
    """
    Observe the grid up to view range cells away.
    """
    def __init__(self, view_range=None, **kwargs):
        super().__init__(**kwargs)
        self.view_range = view_range

    @property
    def view_range(self):
        """
        The number of cells away this agent can observe in each step.
        """
        return self._view_range

    @view_range.setter
    def view_range(self, value):
        assert type(value) is int and 0 <= value, "View range must be a nonnegative integer."
        self._view_range = value

    @property
    def configured(self):
        return super().configured and self.view_range is not None


class MovingAgent(ActingAgent, GridWorldAgent):
    """
    Move up to move_range cells.
    """
    def __init__(self, move_range=None, **kwargs):
        super().__init__(**kwargs)
        self.move_range = move_range

    @property
    def move_range(self):
        """
        The maximum number of cells away that the agent can move.
        """
        return self._move_range

    @move_range.setter
    def move_range(self, value):
        assert type(value) is int and 0 <= value, "Move range must be a nonnegative integer."
        self._move_range = value

    @property
    def configured(self):
        return super().configured and self.move_range is not None


class HealthAgent(GridWorldAgent):
    """
    Agents have health points and can die.

    Health is bounded between 0 and 1. Agents become inactive when the health
    falls to 0.
    """
    def __init__(self, initial_health=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_health = initial_health

    @property
    def health(self):
        """
        The agent's health throughout the simulation trajectory.

        The health will always be between 0 and 1.
        """
        return self._health

    @health.setter
    def health(self, value):
        assert type(value) in [int, float], "Health must be a numeric value."
        self._health = min(max(value, 0), 1)
        self.active = self.health > 0

    @property
    def initial_health(self):
        """
        The agent's initial health between 0 and 1.
        """
        return self._initial_health

    @initial_health.setter
    def initial_health(self, value):
        if value is not None:
            assert type(value) in [int, float], "Initial health must be a numeric value."
            assert 0 < value <= 1, "Initial health must be between 0 and 1."
        self._initial_health = value


class AttackingAgent(ActingAgent, GridWorldAgent):
    """
    Agents that can attack other agents.
    """
    def __init__(self, attack_range=None, attack_strength=None, attack_accuracy=None,
                 simultaneous_attacks=1, **kwargs):
        super().__init__(**kwargs)
        self.attack_range = attack_range
        self.attack_strength = attack_strength
        self.attack_accuracy = attack_accuracy
        self.simultaneous_attacks = simultaneous_attacks

    @property
    def attack_range(self):
        """
        The maximum range of the attack.
        """
        return self._attack_range

    @attack_range.setter
    def attack_range(self, value):
        assert type(value) is int and 0 <= value, "Attack range must be a nonnegative integer."
        self._attack_range = value

    @property
    def attack_strength(self):
        """
        The strength of the attack.

        Should be between 0 and 1.
        """
        return self._attack_strength

    @attack_strength.setter
    def attack_strength(self, value):
        assert type(value) in [int, float], "Attack strength must be a numeric value."
        assert 0 <= value <= 1, "Attack strength must be between 0 and 1."
        self._attack_strength = value

    @property
    def attack_accuracy(self):
        """
        The effective accuracy of the agent's attack.

        Should be between 0 and 1. To make deterministic attacks, use 1.
        """
        return self._attack_accuracy

    @attack_accuracy.setter
    def attack_accuracy(self, value):
        assert type(value) in [int, float], "Attack accuracy must be a numeric value."
        assert 0 <= value <= 1, "Attack accuracy must be between 0 and 1."
        self._attack_accuracy = value

    @property
    def simultaneous_attacks(self):
        """
        The number of attacks the agent can make per turn.

        This parameter is interpreted differently by each attack actor, but generally
        it specifies how many attacks this agent can carry out in a single step.
        See specific AttackActor documentation for more information.
        """
        return self._simultaneous_attacks

    @simultaneous_attacks.setter
    def simultaneous_attacks(self, value):
        assert type(value) is int, "Simultaneous attacks must be an integer."
        assert value >= 0, "Simultaneous attacks must be nonnegative."
        self._simultaneous_attacks = value

    @property
    def configured(self):
        return super().configured and self.attack_range is not None and \
            self.attack_strength is not None and self.attack_accuracy is not None and \
            self.simultaneous_attacks is not None


class AmmoAgent(GridWorldAgent):
    """
    Agent that has a limited amount of ammunition.
    """
    def __init__(self, initial_ammo=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_ammo = initial_ammo

    @property
    def ammo(self):
        """
        The agent's ammo throughout the simulation trajectory.
        """
        return self._ammo

    @ammo.setter
    def ammo(self, value):
        assert type(value) is int, "Ammo must be an integer."
        self._ammo = 0 if value < 0 else value

    @property
    def initial_ammo(self):
        """
        The ammount of ammo with which this agent starts.
        """
        return self._initial_ammo

    @initial_ammo.setter
    def initial_ammo(self, value):
        assert type(value) is int, "Initial ammo must be a an integer."
        self._initial_ammo = value


class AmmoObservingAgentMeta(type):
    """
    AmmoObservingAgentMeta class defines an AmmoObservingAgent as an instance of
    AmmoAgent and ObservingAgent. Then, when we check if an agent is an instance
    of AmmoObservingAgent, it doesn't have to directly derive from it; it just has
    to derive from both AmmoAgent and ObservingAgent.
    """
    def __instancecheck__(self, instance):
        return isinstance(instance, ObservingAgent) and isinstance(instance, AmmoAgent)


class AmmoObservingAgent(AmmoAgent, ObservingAgent, metaclass=AmmoObservingAgentMeta):
    """
    Boilterplate class required to work with the AmmoObserver.
    """
    pass


class OrientationAgent(GridWorldAgent):
    """
    Agent that has an orientation, either 1: Left, 2: Down, 3: Right, 4: Up.
    """
    def __init__(self, initial_orientation=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_orientation = initial_orientation

    @property
    def orientation(self):
        """
        The agent's orientation.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        assert value in range(1, 5), "Orientation must be 1, 2, 3, or 4."
        self._orientation = value

    @property
    def initial_orientation(self):
        """
        The agent's starting orientation at the beginning of the simulation.
        """
        return self._initial_orientation

    @initial_orientation.setter
    def initial_orientation(self, value):
        if value is not None:
            assert value in range(1, 5), "Initial orientation must be 1, 2, 3, or 4."
        self._initial_orientation = value
