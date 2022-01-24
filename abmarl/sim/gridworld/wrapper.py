
from abc import ABC, abstractmethod

from abmarl.sim.gridworld.actor import ActorBaseComponent
from abmarl.sim.gridworld.base import GridWorldBaseComponent

class ComponentWrapper(ABC):
    def __init__(self, component, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(component, GridWorldBaseComponent), \
            "Wrapped component must be a GridWorldBaseComponent."
        self._wrapped_component = component

    @property
    def wrapped_component(self):
        """
        Get the first-level wrapped component.
        """
        return self._wrapped_component

    @property
    def unwrapped(self):
        """
        Fall through all the wrappers and obtain the original, completely unwrapped simulation.
        """
        try:
            return self.wrapped_component.unwrapped
        except AttributeError:
            return self.wrapped_component

    @abstractmethod
    def check_space(self, space):
        pass

    @abstractmethod
    def wrap_space(self, space):
        pass

    @abstractmethod
    def unwrap_space(self, space):
        pass

    @abstractmethod
    def wrap_point(self, space, point):
        pass
    
    @abstractmethod
    def unwrap_point(self, space, point):
        pass

class ActorWrapper(ActorBaseComponent, ComponentWrapper):
    def __init__(self, component, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(component, ActorBaseComponent), \
            "Wrapped component must be an ActorBaseComponent."
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                assert self.check_space(agent.action_space[self.key]), \
                    f"Cannot wrap {self.key} action channel for agent {agent.id}"
                agent.action_space[self.key] = self.wrap_space(agent.action_space[self.key])

    @property
    def key(self):
        return self._wrapped_component.key

    @property
    def supported_agent_type(self):
        return self._wrapped_component.supported_agent_type

    def process_action(self, agent, action_dict, **kwargs):
        if isinstance(agent, self.supported_agent_type):
            action = action_dict[self.key]
            wrapped_action = self.wrap_point(agent.action_space[self.key], action)
            return self.wrapped_component.process_action(
                agent,
                {self.key: wrapped_action},
                **kwargs
            )
