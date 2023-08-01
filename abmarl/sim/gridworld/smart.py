
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
        agents: Dictionary of agents
        grid: The underlying grid. This is typically provided by the builder.
        states: A set of state components. It could be the component class or the
            name of a registered state components.
        observers: A set of observer components. It could be the component class
            or the name of a registered observer component.
        dones: A set of done components. It could be the component class
            or the name of a registered done component.
    """
    def __init__(
            self,
            agents=None,
            grid=None,
            states=None,
            observers=None,
            dones=None,
            **kwargs
    ):

        self.agents = agents
        self.grid = grid

        # State Components
        assert type(states) is set, "States must be a set of state components"
        self._states = set()
        for state in states.items():
            if type(state) is str:
                assert state in registry['state'], f"{state} is not registered as a state."
                self._states.add(registry['state'][state](**kwargs))
            elif issubclass(state, StateBaseComponent):
                self._states.add(state)
            else:
                raise ValueError(
                    f"{state} must be a state component or the name of a registered "
                    "state component."
                )

        # Observer Components
        assert type(observers) is set, "Observers must be a set of observer components"
        self._observers = set()
        for observer in observers.items():
            if type(observer) is str:
                assert observer in registry['observer'], f"{observer} is not registered as an observer."
                self._observers.add(registry['observer'][observer](**kwargs))
            elif issubclass(observer, ObserverBaseComponent):
                self._observers.add(observer)
            else:
                raise ValueError(
                    f"{observer} must be a observer component or the name of a registered "
                    "observer component."
                )

        # Done Components
        assert type(dones) is set, "Dones must be a set of done components"
        self._dones = set()
        for done in dones.items():
            if type(done) is str:
                assert done in registry['done'], f"{done} is not registered as a done component."
                self._dones.add(registry['done'][done](**kwargs))
            elif issubclass(done, DoneBaseComponent):
                self._dones.add(done)
            else:
                raise ValueError(
                    f"{done} must be a done component or the name of a registered "
                    "done component."
                )

    def reset(self, **kwargs):
        for state in self._states:
            state.reset(**kwargs)

        self.rewards = {agent.id: 0 for agent in self.agents.values() if isinstance(agent, Agent)}

    # Note: This is the theoretical approach we could take in stepping the simulation.
    # The first 4 lines are boilerplate and good to have in this super class.
    # After that, the result is unique to the actor. The output may be different, and
    # what the simulation should do with that output is also different. This
    # could be streamlined after we do #337. For now, we leave it up to the subclass
    # to implement the actors and step function.
    # def step(self, action_dict, **kwargs):
    #     if not self._warning_issued:
    #         raise UserWarning("It is best practice to implement your own step function.")
    #     for actor in self._actors:
    #         for agent_id, action in action_dict.items():
    #             agent = self.agents[agent_id]
    #             if agent.active:
    #                 result = actor.process_action(agent, action, **kwargs)
    #                 if result: # Positive result
    #                     self.rewards[agent_id] += 1
    #                 else:
    #                     self.rewards[agent_id] -= 0.1

    #     for agent_id in action_dict:
    #         self.rewards[agent_id] -= 0.01

    def get_obs(self, agent_id, **kwargs):
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
        agent = self.agents[agent_id]
        return all(
            done.get_done(agent, **kwargs) for done in self._dones
        )

    def get_all_done(self, **kwargs):
        return all(
            done.get_all_done(**kwargs) for done in self._dones
        )

    def get_info(self, agent_id, **kwargs):
        return {}