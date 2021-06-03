from abc import ABC, abstractmethod

from admiral.envs import AgentBasedSimulation


class SimulationManager(ABC):
    """
    Control interaction between Trainer and AgentBasedSimulation.

    A Manager implmenents the reset and step API, by which it calls the AgentBasedSimulation
    API, using the getters within reset and step to accomplish the desired control
    flow.

    Attributes:
        env: The AgentBasedSimulation.
        agents: The agents that are in the AgentBasedSimulation.
    """
    def __init__(self, env):
        assert isinstance(env, AgentBasedSimulation), \
            "SimulationManager can only interface with AgentBasedSimulation."
        self.env = env
        self.agents = env.agents

    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the simulation.

        Returns:
            The first obersvation of the agent(s).
        """
        pass

    @abstractmethod
    def step(self, action_dict, **kwargs):
        """
        Step the simulation forward one discrete time-step.

        Args:
            action_dict:
                Dictionary mapping agent(s) to their actions in this time step.

        Returns:
            The observations, rewards, done status, and info for the agent(s) whose
            actions we expect to receive next.

            Note: We do not necessarily return anything for the agent whose actions
            we just received in this time-step.
        """
        pass

    def render(self, **kwargs):
        self.env.render(**kwargs)
