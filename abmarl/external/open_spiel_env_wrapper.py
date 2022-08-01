
from gym.spaces import Discrete
from open_spiel.python.rl_environment import TimeStep, StepType

from abmarl.sim.agent_based_simulation import Agent
from abmarl.managers import TurnBasedManager, SimulationManager


class OpenSpielWrapper:
    """
    Enable connection between SimulationManager and OpenSpiel agents.

    OpenSpiel support turn-based and simultaneous simulations, which Abmarl provides
    through the TurnBasedManager and AllStepManager. OpenSpiel expects TimeStep
    objects as output, which include the observations, rewards, and step type.
    Among the observations, it expects a list of legal actions availbe to the agent.
    The OpenSpielWrapper converts output from the simulation manager to the expected
    format. A TimeStep output typically looks like this:
        TimeStpe(
            observations={
                info_state: {agent_id: agent_obs for agent_id in agents},
                legal_actions: {agent_id: agent_legal_actions for agent_id in agents},
                current_player: current_agent_id
            }
            rewards={
                {agnet_id: agent_reward for agnt_id in agents}
            }
            discounts={
                {agent_id: agent_discout for agent_id in agents}
            }
            step_type=StepType enum
        )

    Furthermore, OpenSpiel provides actions as a list. This wrapper converts it
    those actions to a dict before forwarding it to the underlying simulation manager.

    OpenSpiel does not support the ability for some agents in a simulation to finish
    before others. The simulation is either ongoing, in which all agents are providing
    actions, or else it is done for all agents. In contrast, Abmarl allows some agents to be
    done before others while the simulation is still going. Abmarl expects that done
    agents will not provide actions. OpenSpiel, however, will always provide actions
    for all agents. So this wrapper removes the actions from agents that are
    already done before forwarding the action to the underlying simulation manager.
    Furthermore, OpenSpiel expects every agent to be present in the TimeStep outputs.
    Normally, Abmarl will not provide output for agents that are done since they
    have finished generating data in this episode. In order to work with OpenSpiel,
    this wrapper forces output from all agents at every step.

    Currently, the OpenSpielWrapper only works with simulations in which the action and
    observation space of every agent is Discrete. Most simulations will need to
    be wrapped with the RavelDiscreteWrapper.
    """
    def __init__(self, sim, discounts=1.0, **kwargs):
        assert isinstance(sim, SimulationManager)

        # We keep track of the learning agents separately so that we can append
        # observations and rewards for each of these agents. OpenSpiel expects
        # them to all be present in every time_step.
        self._learning_agents = {
            agent.id: agent for agent in sim.agents.values() if isinstance(agent, Agent)
        }

        # The wrapper assumes that each space is discrete, so we check for that.
        for agent in self._learning_agents.values():
            assert isinstance(agent.observation_space, Discrete) and \
                isinstance(agent.action_space, Discrete), \
                "OpenSpielWrapper can only work with simulations that have all Discrete spaces."
        self.sim = sim

        self.discounts = discounts
        self._should_reset = True

    @property
    def discounts(self):
        """
        The learing discounts for each agent.

        If provided as a number, then that value wil apply to all the agents.
        Make seperate discounts for each agent by providing a dictionary assigning
        each agent to its own discounted value.
        """
        return self._discounts

    @discounts.setter
    def discounts(self, value):
        assert type(value) in (int, float, dict), \
            "The discounts must be either a number or a dict."
        if type(value) in (float, int):
            self._discounts = {
                agent_id: value for agent_id in self._learning_agents
            }
        else: # type(value) == dict
            for discount_id, discount in value.items():
                assert discount_id in self._learning_agents, \
                    "id for the discount must be an agent id."
                assert type(discount) in (float, int), \
                    "discount values must be a number."
            assert all([
                True if agent_id in value.keys() else False for agent_id in self._learning_agents
            ]), "All agents must be given a discounted value."
            self._discounts = value

    @property
    def num_players(self):
        """
        The number of learning agents in the simulation.
        """
        return sum([1 for _ in self._learning_agents])

    @property
    def is_turn_based(self):
        """
        The simulation is turn based if the simulation is wrapped with a TurnBasedManager.
        """
        return isinstance(self.sim, TurnBasedManager)

    @property
    def current_player(self):
        """
        The agent that currently provides the action.

        This is used the in the observation part of the TimeStep output. If it
        is a turn based simulation, then the current player is the single agent who
        is providing an action. If it is a simultaneous simulation, then OpenSpiel does
        not use this property and the current player is just the first agent
        in the list of agents in the simulation.
        """
        return self._current_player

    @current_player.setter
    def current_player(self, value):
        assert value in self._learning_agents, "Current player must be an agent in the simulation."
        self._current_player = value

    def reset(self, **kwargs):
        """
        Reset the simulation.
        """
        self._should_reset = False
        obs = self.sim.reset(**kwargs)
        self.current_player = next(iter(obs))

        observations = {
            "info_state": self._append_obs(obs),
            "legal_actions": {
                agent_id: self.get_legal_actions(agent_id)
                for agent_id in self._learning_agents
            },
            "current_player": self.current_player,
        }

        return TimeStep(
            observations=observations,
            rewards=None,
            discounts=None,
            step_type=StepType.FIRST
        )

    def step(self, action_list, **kwargs):
        """
        Step the simulation forward using the reported actions.

        OpenSpiel provides an action list of either (1) the agent whose turn it
        is in a turn-based simulation or (2) all the agents in a simultaneous simulation. The
        OpenSpielWrapper converts the list of actions to a dictionary before passing
        it to the underlying simulation.

        OpenSpiel does not support the ability for some agents of a simulation to finish
        before others. As such, it may provide actions for agents that are already
        done. To work with Abmarl, this wrapper removes actions for agents that
        are already done.
        """
        # Actions come in as a list, so we need to convert to a dict before forwarding
        # to the SimulationManager.
        if self.is_turn_based:
            action_dict = {self.current_player: action_list[0]}
        else:
            action_dict = {
                agent_id: action_list[i]
                for i, agent_id in enumerate(self._learning_agents)
            }
        # OpenSpiel can send actions for agents that are already done, which doesn't
        # work with our simulation managers. So we filter out these actions before
        # passing them to the manager.
        # TODO: Although all implemeted managers do track the done agents, this is not
        # a part of the SimulationManager interface.
        for agent_id in self.sim.done_agents:
            try:
                del action_dict[agent_id]
            except KeyError:
                pass
        # We have just deleted actions from agents that are already done, which
        # can result in an empty action dictionary (e.g. in a turn-based simulation).
        # In this case, we just take a fake step.
        if not action_dict: # No actions
            return self._take_fake_step()

        if self._should_reset:
            return self.reset(**kwargs)

        obs, reward, done, info = self.sim.step(action_dict, **kwargs)
        self.current_player = next(iter(obs))

        step_type = StepType.LAST if done['__all__'] else StepType.MID
        self._should_reset = step_type == StepType.LAST

        observations = {
            "info_state": self._append_obs(obs),
            "legal_actions": {
                agent_id: self.get_legal_actions(agent_id)
                for agent_id in self._learning_agents
            },
            "current_player": self.current_player,
        }

        return TimeStep(
            observations=observations,
            rewards=self._append_reward(reward),
            discounts=self._discounts,
            step_type=step_type
        )

    def observation_spec(self):
        """
        The agents' observations spaces.

        Abmarl uses gym spaces for the observation space. This wrapper converts
        the gym space into a format that OpenSpiel expects.
        """
        return {
            agent.id: {
                'info_state': (agent.observation_space.n,),
                'legal_actions': (agent.action_space.n,),
                'current_player': ()
            } for agent in self._learning_agents.values()
        }

    def action_spec(self):
        """
        The agents' action spaces.

        Abmarl uses gym spaces for the action space. This wrapper converts
        the gym space into a format that OpenSpiel expects.
        """
        return {
            agent.id: {
                'num_actions': agent.action_space.n,
                'min': 0,
                'max': agent.action_space.n - 1,
                'dtype': int
            } for agent in self._learning_agents.values()
        }

    def get_legal_actions(self, agent_id):
        """
        Return the legal actions available to the agent.

        By default, this wrapper uses all the available actions as the legal actions
        in each time step. This function can be overwritten in a derived class
        to add logic for obtaining the actual legal actions available.
        """
        return [i for i in range(self._learning_agents[agent_id].action_space.n)]

    def _append_obs(self, obs):
        # OpenSpiel expects every agent to appear in the observation at every
        # time step. The simulation manager won't produce an observation for a
        # done agent, so we have to add it ourselves.
        for agent_id in self._learning_agents:
            if agent_id not in obs:
                obs[agent_id] = self.sim.sim.get_obs(agent_id)
        return obs

    def _append_reward(self, reward):
        # OpenSpiel expects every agent to appear in the reward at every
        # time step. The simulation manager won't produce a reward for a
        # done agent, so we have to add it ourselves.
        for agent_id in self._learning_agents:
            if agent_id not in reward:
                reward[agent_id] = 0
        return reward

    def _take_fake_step(self):
        # This is used when all the actions are from done agents. In that case,
        # we just move along with no state update.
        obs = self._append_obs({})
        self.current_player = next(iter(obs))
        observations = {
            "info_state": obs,
            "legal_actions": {
                agent_id: self.get_legal_actions(agent_id)
                for agent_id in self._learning_agents
            },
            "current_player": self.current_player,
        }
        return TimeStep(
            observations=observations,
            rewards=self._append_reward({}),
            discounts=self._discounts,
            step_type=StepType.MID
        )
