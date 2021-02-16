
from .agent_based_simulation import AgentBasedSimulation
from admiral.envs.components.agent import Agent, ObservingAgent, ActingAgent

class SimpleAgent(ObservingAgent, ActingAgent):
    """
    A SimpleAgent that observes and acts for use with environments that don't use
    components.
    """
    pass

