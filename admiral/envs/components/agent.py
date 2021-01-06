
# ------------------ #
# --- Base Agent --- #
# ------------------ #

class Agent:
    """
    Base Agent class for agents that live in an environment. Agents require an
    id in in order to even be constructed. Agents must also have an observation
    space and action space to be considered successfully configured.
    """
    def __init__(self, id=None, observation_space=None, action_space=None, **kwargs):
        if id is None:
            raise TypeError("Agents must be constructed with an id.")
        else:
            self.id = id
        self.observation_space = {} if observation_space is None else observation_space
        self.action_space = {} if action_space is None else action_space
    
    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return self.id is not None and self.action_space and self.observation_space

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ if isinstance(other, self.__class__) else False



# ----------------- #
# --- Attacking --- #
# ----------------- #

class AttackingAgent(Agent):
    """
    Agents that can attack other agents.

    attack_range (int):
        The effective range of the attack. Can be used to determine if an attack
        is successful based on distance between agents.
    
    attack_strength (float):
        How effective the agent's attack is. This is applicable in situations where
        the agents' health is affected by attacks.
    """
    def __init__(self, attack_range=None, attack_strength=None, **kwargs):
        super().__init__(**kwargs)
        assert attack_range is not None, "attack_range must be a nonnegative integer"
        self.attack_range = attack_range
        assert attack_strength is not None, "attack_strength must be a nonnegative number"
        self.attack_strength = attack_strength
    
    @property
    def configured(self):
        """
        The agent is successfully configured if the attack range and strength is
        specified.
        """
        return super().configured and self.attack_range is not None and self.attack_strength is not None



# ----------------------- #
# --- Health and Life --- #
# ----------------------- #

class LifeAgent(Agent):
    """
    Agents have health and are alive or dead.

    min_health (float):
        The minimum value the health can reach before the agent dies.
    
    max_health (float):
        The maximum value the health can reach.

    initial_health (float):
        The initial health of the agent. The health will be set to this initial
        option at reset time.
    """
    def __init__(self, min_health=0.0, max_health=1.0, initial_health=None, **kwargs):
        super().__init__(**kwargs)
        if initial_health is not None:
            assert min_health <= initial_health <= max_health
        self.initial_health = initial_health
        self.min_health = min_health
        self.max_health = max_health
        self.is_alive = True
        self.health = None

    @property
    def configured(self):
        """
        The agent is successfully configured if the min and max health are specified
        and if is_alive is specified.
        """
        return super().configured and self.min_health is not None and self.max_health is not None and self.is_alive is not None



# ----------------- #
# --- Observing --- #
# ----------------- #

class AgentObservingAgent(Agent):
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

class ResourceObservingAgent(Agent):
    """
    Agents can observe the resources in the environment.

    resource_view_range (int):
        Any resources within this range of the agent's position will be fully observed.
    """
    def __init__(self, resource_view_range=None, **kwargs):
        super().__init__(**kwargs)
        assert resource_view_range is not None, "resource_view_range must be nonnegative integer"
        self.resource_view_range = resource_view_range
    
    @property
    def configured(self):
        """
        Agents are configured if the resource_view_range parameter is set.
        """
        return super().configured and self.resource_view_range is not None



# ----------------------------- #
# --- Position and Movement --- #
# ----------------------------- #

class PositionAgent(Agent):
    """
    Agents have a position in the environment.

    starting_position (np.array):
        The desired starting position for this agent.
    """
    def __init__(self, starting_position=None, **kwargs):
        super().__init__(**kwargs)
        self.starting_position = starting_position
        self.position = None

class GridMovementAgent(Agent):
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



# -------------------------------- #
# --- Resources and Harvesting --- #
# -------------------------------- #

class HarvestingAgent(Agent):
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



# ------------ #
# --- Team --- #
# ------------ #

class TeamAgent(Agent):
    """
    Agents are on a team, which will affect their ability to perform certain actions,
    such as who they can attack.
    """
    def __init__(self, team=None, **kwargs):
        super().__init__(**kwargs)
        assert team is not None, "team must be an integer"
        self.team = team
    
    @property
    def configured(self):
        """
        Agent is configured if team is set.
        """
        return super().configured and self.team is not None
