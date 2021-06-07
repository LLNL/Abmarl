from abc import ABC, abstractmethod

from abmarl.sim import AgentBasedSimulation


class SimulationManager(ABC):
    """
    Control interaction between Trainer and AgentBasedSimulation.

    A Manager implmenents the reset and step API, by which it calls the AgentBasedSimulation
    API, using the getters within reset and step to accomplish the desired control
    flow.

    Attributes:
        sim: The AgentBasedSimulation.
        agents: The agents that are in the AgentBasedSimulation.
    """
    def __init__(self, sim):
        assert isinstance(sim, AgentBasedSimulation), \
            "SimulationManager can only interface with AgentBasedSimulation."
        self.sim = sim
        self.agents = sim.agents

    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the simulation.

        Returns:
            The first observation of the agent(s).
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
            we just received in this time-step. This behavior is defined by each
            Manager.
        """
        pass

    def render(self, **kwargs):
        self.sim.render(**kwargs)
