
from gym.spaces import Dict
from ray.rllib import MultiAgentEnv

from abmarl.sim.agent_based_simulation import ActingAgent, Agent, ObservingAgent


class MultiAgentWrapper(MultiAgentEnv):
    """
    Enable connection between SimulationManager and RLlib Trainer.

    Wraps a SimulationManager and forwards all calls to the manager. This class
    is boilerplate and needed because RLlib checks that the simulation is an instance
    of MultiAgentEnv.

    Attributes:
        sim: The SimulationManager.
    """
    def __init__(self, sim):
        from abmarl.managers import SimulationManager
        assert isinstance(sim, SimulationManager)
        self.sim = sim

        self._agent_ids = set(
            agent.id for agent in self.sim.agents.values()
            if isinstance(agent, Agent)
        )
        self.observation_space = Dict({
            agent.id: agent.observation_space
            for agent in self.sim.agents.values()
            if isinstance(agent, ObservingAgent)
        })
        self.action_space = Dict({
            agent.id: agent.action_space
            for agent in self.sim.agents.values()
            if isinstance(agent, ActingAgent)
        })

    def reset(self):
        """See SimulationManager."""
        return self.sim.reset()

    def step(self, actions):
        """See SimulationManager."""
        return self.sim.step(actions)

    def render(self, *args, **kwargs):
        """See SimulationManager."""
        return self.sim.render(*args, **kwargs)
