
from gym.spaces import Discrete
from open_spiel.python.rl_environment import TimeStep, StepType

from abmarl.sim.agent_based_simulation import Agent
from abmarl.managers import TurnBasedManager, SimulationManager

class OpenSpielWrapper:
    def __init__(self, sim, discount=1.0, **kwargs):
        assert isinstance(sim, SimulationManager)
        # The wrapper assumes that each space is discrete, so we should check for
        # that.
        for agent in self.sim.agents.values():
            assert isinstance(agent.observation_space, Discrete) and \
                isinstance(agent.action_space, Discrete), \
                "OpenSpielWrapper can only work with simulations that have all Discrete spaces."
        self.sim = sim
        if type(discount) is float:
            discount = {
                agent.id: discount
                for agent in self.sim.agents.values() if isinstance(agent, Agent)
            }
        self._discounts = discount
        self._should_reset = True

    @property
    def num_players(self):
        return sum([1 for agent in self.sim.agents.values() if isinstance(agent, Agent)])

    @property
    def is_turn_based(self):
        return isinstance(self.sim, TurnBasedManager)


    def reset(self, **kwargs):
        self._should_reset = False
        obs = self.sim.reset(**kwargs)

        # If it is a turn based sim, then the current player should be the next
        # player whose turn it will be. But what is current player for an all-step game?
        observations = {
            "info_state": obs,
            "legal_actions": {
                agent.id: self.get_legal_actions(agent.id)
                for agent in self.sim.agents.values() if isinstance(agent, Agent)
            },
            "current_player": next(iter(obs)),
        }

        return TimeStep(
            observations=observations,
            rewards=None,
            discounts=None,
            step_type=StepType.FIRST
        )

    def step(self, actions, **kwargs):
        # TODO: Do actions come in as a list or a dict?
        if self._should_reset:
            return self.reset(**kwargs)

        obs, reward, done, info = self.sim.step(actions, **kwargs)

        step_type = StepType.LAST if done else StepType.MID
        self._should_reset = step_type == StepType.LAST

        observations = {
            "info_state": obs,
            "legal_actions": {
                agent.id: self.get_legal_actions(agent.id)
                for agent in self.sim.agents.values() if isinstance(agent, Agent)
            },
            "current_player": next(iter(obs)),
        }

        return TimeStep(
            observations=observations,
            rewards=reward,
            discounts=self._discounts,
            step_type=step_type
        )

    def observation_spec(self):
        return {
            agent.id: {
                'info_state': (agent.observation_space.n,),
                'legal_actions': (agent.action_space.n,),
                'current_player': ()
            } for agent in self.sim.agents.values() if isinstance(agent, Agent)
        }

    def action_spec(self):
        return {
            agent.id: {
                'num_actions': agent.action_space.n,
                'min': 0,
                'max': agent.action_space.n - 1,
                'dtype': int
            } for agent in self.sim.agents.values() if isinstance(agent, Agent)
        }

    def get_legal_actions(self, agent_id):
        """
        Return the legal actions available to the player.

        By default, this wrapper uses all the available actions as the legal actions
        in each time step. This function can be overwritten in a derived class
        to add logic for obtaining the actual legal actions available.
        """
        return [self.sim.agents[agent_id].action_space.n]
