from abc import ABC, abstractmethod

from abmarl.tools import gym_utils as gu


class PrincipleAgent:
    """
    Principle Agent class for agents in a simulation.
    """
    def __init__(self, id=None, seed=None, **kwargs):
        self.id = id
        self.seed = seed

    @property
    def id(self):
        """The agent's unique identifier."""
        return self._id

    @id.setter
    def id(self, value):
        assert type(value) is str, "id must be a string."
        self._id = value

    @property
    def seed(self):
        """Seed for random number generation."""
        return self._seed

    @seed.setter
    def seed(self, value):
        assert value is None or type(value) is int, "Seed must be an integer."
        self._seed = value

    @property
    def active(self):
        """
        True if the agent is still active in the simulation.

        Active means that the agent is in a valid state. For example, suppose agents
        in our Simulation can die. Then active is True if the agents are alive
        or False if they're dead.
        """
        return True

    @property
    def configured(self):
        """
        All agents must have an id.
        """
        return self.id is not None

    def finalize(self, **kwargs):
        pass

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ if isinstance(other, self.__class__) else False


class ActingAgent(PrincipleAgent):
    """
    ActingAgents can act in the simulation.

    The Trainer will produce actions for the agents and send them to the SimulationManager,
    which will process those actions in its step function.
    """
    def __init__(self, action_space=None, **kwargs):
        super().__init__(**kwargs)
        self.action_space = action_space

    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, value):
        assert value is None or gu.check_space(value), \
            "The action space must be None, a gym Space, or a dict of gym Spaces."
        self._action_space = {} if value is None else value

    @property
    def configured(self):
        """
        Acting agents must have an action space.
        """
        return super().configured and gu.check_space(self.action_space, strict=True)

    def finalize(self, **kwargs):
        """
        Wrap all the action spaces with a Dict if applicable and seed it if the agent was
        created with a seed.
        """
        super().finalize(**kwargs)
        if type(self.action_space) is dict:
            self.action_space = gu.make_dict(self.action_space)
        self.action_space.seed(self.seed)


class ObservingAgent(PrincipleAgent):
    """
    ObservingAgents can observe the state of the simulation.

    The agent's observation must be *in* its observation space. The SimulationManager
    will send the observation to the Trainer, which will use it to produce actions.
    """
    def __init__(self, observation_space=None, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = observation_space

    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        assert value is None or gu.check_space(value), \
            "The observation space must be None, a gym Space, or a dict of gym Spaces."
        self._observation_space = {} if value is None else value

    @property
    def configured(self):
        """
        Observing agents must have an observation space.
        """
        return super().configured and gu.check_space(self.observation_space, strict=True)

    def finalize(self, **kwargs):
        """
        Wrap all the observation spaces with a Dict and seed it if the agent was
        created with a seed.
        """
        super().finalize(**kwargs)
        if type(self.observation_space) is dict:
            self.observation_space = gu.make_dict(self.observation_space)
        self.observation_space.seed(self.seed)


class Agent(ObservingAgent, ActingAgent):
    """
    An Agent that can both observe and act.
    """
    pass


class AgentBasedSimulation(ABC):
    """
    AgentBasedSimulation interface.

    Under this design model the observations, rewards, and done conditions of the
    agents is treated as part of the simulations internal state instead of as
    output from reset and step. Thus, it is the simulations responsibility to manage
    rewards and dones as part of its state (e.g. via self.rewards dictionary).

    This interface supports both single- and multi-agent simulations by treating
    the single-agent simulation as a special case of the multi-agent, where there
    is only a single agent in the agents dictionary.
    """

    @property
    def agents(self):
        """
        A dict that maps the Agent's id to the Agent object. An Agent must be an
        instance of PrincipleAgent. A multi-agent simulation is expected to have
        multiple entries in the dictionary, whereas a single-agent simulation
        should only have a single entry in the dictionary.
        """

        return self._agents

    @agents.setter
    def agents(self, value_agents):
        assert type(value_agents) is dict, "Agents must be a dict."
        for agent_id, agent in value_agents.items():
            assert isinstance(agent, PrincipleAgent), \
                "Values of agents dict must be instance of PrincipleAgent."
            assert agent_id == agent.id, \
                "Keys of agents dict must be the same as the Agent's id."
        self._agents = value_agents

    def finalize(self):
        """
        Finalize the initialization process. At this point, every agent should
        be configured with action and observation spaces, which we convert into
        Dict spaces for interfacing with the trainer.
        """
        for agent in self.agents.values():
            agent.finalize()
            assert agent.configured

    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the simulation simulation to a start state, which may be randomly
        generated.
        """
        pass

    @abstractmethod
    def step(self, action, **kwargs):
        """
        Step the simulation forward one discrete time-step. The action is a dictionary
        that contains the action of each agent in this time-step.
        """
        pass

    @abstractmethod
    def render(self, **kwargs):
        """
        Render the simulation for vizualization.
        """
        pass

    @abstractmethod
    def get_obs(self, agent_id, **kwargs):
        """
        Return the agent's observation.
        """
        pass

    @abstractmethod
    def get_reward(self, agent_id, **kwargs):
        """
        Return the agent's reward.
        """
        pass

    @abstractmethod
    def get_done(self, agent_id, **kwargs):
        """
        Return the agent's done status.
        """
        pass

    @abstractmethod
    def get_all_done(self, **kwargs):
        """
        Return the simulation's done status.
        """
        pass

    @abstractmethod
    def get_info(self, agent_id, **kwargs):
        """
        Return the agent's info.
        """
        pass
