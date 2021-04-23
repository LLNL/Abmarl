
from abc import ABC, abstractmethod

class Agent:
    """
    Base Agent class for agents that live in an environment. All agents require
    a string id.

    id (str):
        The agent's id.

    seed (int):
        Seed this agent's rng. Default value is None.
    """
    def __init__(self, id=None, seed=None, **kwargs):
        self.id = id
        self.seed = seed
    
    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, value):
        assert type(value) is str, "id must be a string."
        self._id = value
    
    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, value):
        assert value is None or type(value) is int, "Seed must be an integer."
        self._seed = value

    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return self.id is not None
    
    def finalize(self, **kwargs):
        pass

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ if isinstance(other, self.__class__) else False

class ActingAgent(Agent):
    """
    ActingAgents are Agents that are expected to produce actions and therefore
    should have an action space in order to be successfully configured.
    """
    def __init__(self, action_space=None, **kwargs):
        super().__init__(**kwargs)
        self.action_space = action_space
    
    @property
    def action_space(self):
        return self._action_space
    
    @action_space.setter
    def action_space(self, value):
        self._action_space = {} if value is None else value
    
    @property
    def configured(self):
        return super().configured and self.action_space is not None
            
    def finalize(self, **kwargs):
        """
        Wrap all the action spaces with a Dict if applicable and seed it if the agent was
        created with a seed.
        """
        super().finalize(**kwargs)
        if type(self.action_space) is dict:
            from gym.spaces import Dict
            self.action_space = Dict(self.action_space)
        self.action_space.seed(self.seed)

class ObservingAgent(Agent):
    """
    ObservingAgents are Agents that are expected to receive observations and therefore
    should have an observation space in order to be successfully configured.
    """
    def __init__(self, observation_space=None, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = observation_space

    @property
    def observation_space(self):
        return self._observation_space
    
    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = {} if value is None else value
    
    @property
    def configured(self):
        return super().configured and self.observation_space is not None

    def finalize(self, **kwargs):
        """
        Wrap all the observation spaces with a Dict and seed it if the agent was
        created with a seed.
        """
        super().finalize(**kwargs)
        if type(self.observation_space) is dict:
            from gym.spaces import Dict
            self.observation_space = Dict(self.observation_space)
        self.observation_space.seed(self.seed)



class AgentBasedSimulation(ABC):
    """
    AgentBasedSimulation defines the interface that agent-based simulations will
    implement. The interface defines the following API:
        agents: dict
            A dictionary that maps the Agent's id to Agent object. An Agent object can hold
            any property, but it must include an id, the action space, and the
            observation space. A multi-agent environment is expected to have
            multiple entries in the dictionary, whereas a single-agent environment
            should only have a single entry in the dictionary.
        reset()
            Reset the simulation environment to a start state, which may be randomly
            generated.
        step(action: dict)
            Step the environment forward one discrete time-step. The action is a dictionary that
            contains the action of each agent in this time-step.
        render
            Render the enviroment for visualization.
        get_obs(agent_id) -> obj
            Get the observation of the respective agent.
        get_reward(agent_id) -> double
            Get the reward for the respective agent.
        get_done(agent_id) -> bool
            Get the done status of the respective agent.
        get_all_done() -> bool
            Get the done status for all the agents and/or the environment.
        get_info(agent_id) -> dict
            Get additional information that can be used for analysis or debugging.
    Under this design model the observations, rewards, and done conditions of the
    agents is treated as part of the environments internal state instead of as
    output from reset and step. Thus, it is the environments responsibility to manage
    rewards and dones as part of its state (e.g. via self.rewards dictionary).
    
    This interface supports both single- and multi-agent environments by treating
    the single-agent environment as a special case of the multi-agent, where there
    is only a single agent in the agents dictionary.
    """
    @property
    def agents(self):
        return self._agents

    @agents.setter
    def agents(self, value_agents):
        assert type(value_agents) is dict, "Agents must be a dictionary."
        for agent_id, agent in value_agents.items():
            assert isinstance(agent, Agent), "Values of agents dictionary must be of type Agent."
            assert agent_id == agent.id, "Keys of the dictionary must be the same as the Agent's id."
        self._agents = value_agents

    def finalize(self):
        """
        Finalize the initialization process. At this point, every agent should
        be configured with action and observation spaces, which we convert into
        Dict spaces for interfacing with the trainer.
        """
        for agent in self.agents.values():
            assert agent.configured
            agent.finalize()
    
    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the simulation environment.
        """
        pass

    @abstractmethod
    def step(self, action, **kwargs):
        """
        Step the environment forward one discrete time-step. The action is a dictionary
        that contains the action of each agent in this time-step.
        """
        pass

    @abstractmethod
    def render(self, **kwargs):
        """
        Render the environment for vizualization.
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
        Return the environment's done status.
        """
        pass
    
    @abstractmethod
    def get_info(self, agent_id, **kwargs):
        """
        Return the agent's info.
        """
        pass
