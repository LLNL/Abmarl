from abc import ABC, abstractmethod


class SimulationManager(ABC):
    """
    The SimulationManager defines the interface for controlling the interaction
    between the AgentBasedSimulations and the episode generator. A Manager implmenents
    the reset and step API, by which it calls the AgentBasedSimulation API, using the
    getters within reset and step to accomplish the desired control flow.
        reset() -> obj
            Reset the simulation environment to a start state, which may be randomly
            generated. Return the first observation of the agent(s).
        step(action: dict) -> obj, double, bool, dict
            Step the environment forward one discrete time-step. The action is a dictionary that
            contains the action of each agent in this time-step. The return is the observation,
            reward, done status, and extra information for the respective agents.
        render
            Render the enviroment for visualization.
    """
    def __init__(self, env):
        from admiral.envs import AgentBasedSimulation
        assert isinstance(env, AgentBasedSimulation), \
            "SimulationManager can only interface with AgentBasedSimulation."
        self.env = env
        self.agents = env.agents

    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the environment and return an observation.
        """
        pass

    @abstractmethod
    def step(self, action_dict, **kwargs):
        """
        Step the environment according to the indicated actions and return the
        observation, reward, done, and info.
        """
        pass

    def render(self, **kwargs):
        self.env.render(**kwargs)
