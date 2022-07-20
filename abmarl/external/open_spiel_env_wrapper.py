
from open_spiel.python.rl_environment import TimeStep, StepType

from abmarl.sim.agent_based_simulation import Agent
from abmarl.managers import TurnBasedManager, SimulationManager

class OpenSpielWrapper:
    def __init__(self, sim, **kwargs):
        assert isinstance(sim, SimulationManager)
        self.sim = sim
        self._should_reset = True

    @property
    def num_players(self):
        pass

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
            "current_player": 0, # TODO: figure out the logic for this!
        }

        return TimeStep(
            observations=observations,
            rewards=None,
            discounts=None,
            step_type=StepType.FIRST
        )

    def step(self, actions, **kwargs):
        if self._should_reset:
            return self.reset(**kwargs)

        obs, reward, done, info = self.sim.step(actions, **kwargs)

        step_type = StepType.LAST if done else StepType.MID
        self._should_reset = step_type == StepType.LAST

        # observations = {
        #     "info_state": obs,
        #     "legal_actions": self.get_legal_actions(),
        #     "current_player": 0, # TODO: figure out the logic for this!
        # }

        observations = {
            "info_state": obs,
            "legal_actions": {
                agent.id: self.get_legal_actions(agent.id)
                for agent in self.sim.agents.values() if isinstance(agent, Agent)
            },
            "current_player": 0, # TODO: figure out the logic for this!
        }

        return TimeStep(
            observations=observations,
            rewards=[reward],
            discounts=self._discounts, # TODO: I need to get discounts
            step_type=step_type
        )

    def observation_spec(self):
        pass

    def action_spec(self):
        pass

    def get_legal_actions(self, agent_id):
        """
        Return the legal actions available to the player.

        By default, this wrapper uses all the available actions as the legal actions
        in each time step. This function can be overwritten in a derived class
        to add logic for obtaining the actual legal actions available.
        """
        return [self.sim.agents[agent_id].action_space.n] # TODO: Something like this.

    # def get_legal_actions(self):
    #     """
    #     Return the legal actions available to the player.

    #     By default, this wrapper uses all the available actions as the legal actions
    #     in each time step. This function can be overwritten in a derived class
    #     to add logic for obtaining the actual legal actions available.
    #     """
    #     return {
    #         agent.id: [agent.action_space.n] # TODO: Something like this
    #         for agent in self.sim.agents.values() if isinstance(agent, Agent)
    #     }

