
from gymnasium.spaces import Dict

from abmarl.sim.agent_based_simulation import ActingAgent, ObservingAgent, Agent, \
    is_agent, AgentBasedSimulation

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
                if is_agent(agent)
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

        def reset(self, *args, **kwargs):
            """See SimulationManager."""
            return self.sim.reset(), {}

        def step(self, actions, *args, **kwargs):
            """See SimulationManager."""
            obs, rewards, dones, infos = self.sim.step(actions)
            return obs, rewards, dones, {"__all__": False}, infos

        def render(self, *args, **kwargs):
            """See SimulationManager."""
            return self.sim.render(*args, **kwargs)


    class MultiAgentABS(AgentBasedSimulation):
        """
        Wraps an RLlib MultiAgentEnv and leverages it for implementing the ABS interface.
        """
        def __init__(self, multi_agent_env, null_observation=None, null_action=None, **kwargs):
            assert isinstance(multi_agent_env, MultiAgentEnv), \
                "multi_agent_env must be a MultiAgentEnv."
            assert multi_agent_env._action_space_in_preferred_format and \
                multi_agent_env._obs_space_in_preferred_format, \
                "The action and observation spaces must be in the preferred format."
            self._env = multi_agent_env
            if not null_action:
                null_action = {}
            if not null_observation:
                null_observation = {}
            agents = {
                agent_id: Agent(
                    id=agent_id,
                    observation_space=multi_agent_env.observation_space[agent_id],
                    null_observation=null_observation.get(agent_id),
                    action_space=multi_agent_env.action_space[agent_id],
                    null_action=null_action.get(agent_id),
                ) for agent_id in multi_agent_env._agent_ids
            }
            super().__init__(agents=agents, **kwargs)
            # ABS storage
            self._obs = None
            self._reward = None
            self._done = None
            self._info = None

        def reset(self, **kwargs):
            """
            Reset the simulation and store the observation and info.
            """
            self._obs, self._info = self._env.reset()

        def step(self, action_dict, *args, **kwargs):
            """
            Step the simulation and store the relevant data.
            """
            self._obs, self._reward, term, trunc, self._info = self._env.step(
                action_dict, *args, **kwargs
            )
            self._done = {**term, **trunc}
            for agent in self._done:
                self._done[agent] = term.get(agent, False) or trunc.get(agent, False)

        def render(self, **kwargs):
            self._env.render(**kwargs)

        def get_obs(self, *args, **kwargs):
            """
            Return the stored observation, either from reset or step, whichever was last called.
            """
            return self._obs

        def get_reward(self, *args, **kwargs):
            """
            Return the stored reward, either from reset or step, whichever was last called.
            """
            return self._reward

        def get_done(self, *args, **kwargs):
            """
            Return the stored done status, either from reset or step, whichever was last called.
            """
            return self._done

        def get_all_done(self, **kwargs):
            """
            Same thing as get done.
            """
            return self._done

        def get_info(self, *args, **kwargs):
            """
            Return the stored info, either from reset or step, whichever was last called.
            """
            return self._info


    def multi_agent_to_abmarl(
            multi_agent_env,
            null_observation=None,
            null_action=None,
            ):
        return MultiAgentABS(
            multi_agent_env,
            null_observation,
            null_action
        )

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

    class MultiAgentABS(AgentBasedSimulation):
        """
        Stub for MultiAgentABS class, which is not implemented without RLlib.
        """
        def __init__(self, sim):
            raise NotImplementedError(
                "Cannot use MultiAgentABS without RLlib. Please install the "
                "RLlib extra with, for example, pip install abmarl[rllib]."
            )

    def multi_agent_to_abmarl(*args, **kwargs):
        NotImplementedError(
            "Cannot use multi_agent_to_abmarl without RLlib. Please install the "
            "RLlib extra with, for example, pip install abmarl[rllib]."
        )
