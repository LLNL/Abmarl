
from gym.spaces import Dict

from abmarl.sim.agent_based_simulation import ActingAgent, Agent, ObservingAgent

try:
    from ray.rllib import MultiAgentEnv

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
            self._spaces_in_preferred_format = True

        @property
        def unwrapped(self):
            """
            Fall through all the wrappers and obtain the original, completely unwrapped simulation.
            """
            try:
                return self.sim.unwrapped
            except AttributeError:
                return self.sim

        def reset(self):
            """See SimulationManager."""
            return self.sim.reset()

        def step(self, actions):
            """See SimulationManager."""
            return self.sim.step(actions)

        def render(self, *args, **kwargs):
            """See SimulationManager."""
            return self.sim.render(*args, **kwargs)

except ImportError:
    class MultiAgentWrapper:
        """
        Stub for MultiAgentWrapper class, which is not implemented without RLlib.
        """
        def __init__(self, sim):
            raise NotImplementedError(
                "Cannot use MultiAgentWrapper without RLlib. Please install the "
                "RLlib extra with, for example, pip install abmarl[rllib]."
            )
