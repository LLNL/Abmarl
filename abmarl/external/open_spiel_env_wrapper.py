
from open_spiel.python.rl_environment import TimeStep, StepType

from abmarl.managers import TurnBasedManager

class OpenSpielWrapper:
    def __init__(self, sim, **kwargs):
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

        observations = {
            "info_state": [obs],
            "legal_actions": [legal_actions], # TODO: Get this!
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
        legal_actions = [] # TODO: Build this up!

        observations = {
            "info_state": [obs],
            "legal_actions": [legal_actions],
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

