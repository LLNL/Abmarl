
from abc import ABC

from abmarl.sim.agent_based_simulation import Agent
from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.registry import registry, DoneBaseComponent, \
    ObserverBaseComponent, StateBaseComponent


class SmartGridWorldSimulation(GridWorldSimulation, ABC):
    """
    Default "template" for building and running simulations.

    The SmartGridWorldSimulation supports varying some components of a simluation
    at initialzation without changing simulation code. Actor components and the
    step function must still be implemented by the sub class.

    Args:
        states: A set of state components. It could be the component class or the
            name of a registered state components.
        observers: A set of observer components. It could be the component class
            or the name of a registered observer component.
        dones: A set of done components. It could be the component class
            or the name of a registered done component.
    """
    def __init__(
            self,
            states=None,
            observers=None,
            dones=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        # State Components
        if states:
            assert type(states) is set, "States must be a set of state components"
            self._states = set()
            for state in states:
                if type(state) is str:
                    assert state in registry['state'], f"{state} is not registered as a state."
                    self._states.add(
                        registry['state'][state](**kwargs)
                    )
                elif issubclass(state, StateBaseComponent):
                    self._states.add(state(**kwargs))
                else:
                    raise ValueError(
                        f"{state} must be a state component or the name of a registered "
                        "state component."
                    )

        # Observer Components
        if observers:
            assert type(observers) is set, "Observers must be a set of observer components"
            self._observers = set()
            for observer in observers:
                if type(observer) is str:
                    assert observer in registry['observer'], \
                        f"{observer} is not registered as an observer."
                    self._observers.add(registry['observer'][observer](**kwargs))
                elif issubclass(observer, ObserverBaseComponent):
                    self._observers.add(observer(**kwargs))
                else:
                    raise ValueError(
                        f"{observer} must be a observer component or the name of a registered "
                        "observer component."
                    )

        # Done Components
        if dones:
            assert type(dones) is set, "Dones must be a set of done components"
            self._dones = set()
            for done in dones:
                if type(done) is str:
                    assert done in registry['done'], \
                        f"{done} is not registered as a done component."
                    self._dones.add(registry['done'][done](**kwargs))
                elif issubclass(done, DoneBaseComponent):
                    self._dones.add(done(**kwargs))
                else:
                    raise ValueError(
                        f"{done} must be a done component or the name of a registered "
                        "done component."
                    )

    def reset(self, **kwargs):
        assert hasattr(self, '_states'), "Smart Simulation requires '_states' attribute."
        for state in self._states:
            state.reset(**kwargs)

        self.rewards = {agent.id: 0 for agent in self.agents.values() if isinstance(agent, Agent)}

    def get_obs(self, agent_id, **kwargs):
        assert hasattr(self, '_observers'), "Smart Simulation requires '_observers' attribute."
        agent = self.agents[agent_id]
        return {
            k: v for observer in self._observers
            for k, v in observer.get_obs(agent, **kwargs).items()
        }

    def get_reward(self, agent_id, **kwargs):
        reward = self.rewards[agent_id]
        self.rewards[agent_id] = 0
        return reward

    def get_done(self, agent_id, **kwargs):
        assert hasattr(self, '_dones'), "Smart Simulation requires '_dones' attribute."
        agent = self.agents[agent_id]
        return all(
            done.get_done(agent, **kwargs) for done in self._dones
        )

    def get_all_done(self, **kwargs):
        assert hasattr(self, '_dones'), "Smart Simulation requires '_dones' attribute."
        return all(
            done.get_all_done(**kwargs) for done in self._dones
        )

    def get_info(self, agent_id, **kwargs):
        return {}
