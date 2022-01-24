
from abc import abstractmethod

from abmarl.sim.gridworld.actor import ActorBaseComponent
from abmarl.sim.gridworld.base import GridWorldBaseComponent

# TODO: Should this be a GridWorldBaseComponent?
# The ActorWrapper is already a GridWorldBaseComponent, so we are getting
# the interface. This raises a bigger question about the design because the
# Wrapper doesn't necessarily see the agents and the grid. It should just use
# that from the component that it wraps. So we need a way to specify that the
# wrapper does implement the api of the component, but that it doesn't receive
# its own reference to the grid or the agent.
# Instead of the super().__init__(), we can specify to only call the wrapper parent
# chain instead of the component's init.
class ComponentWrapper:
    """
    Wraps GridWorldBaseComponent.
    """
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
        """
        Verify that the space can be wrapped.
        """
        pass

    @abstractmethod
    def wrap_space(self, space):
        """
        Wrap the space.

        Args:
            space: The space to wrap.
        """
        pass

    @abstractmethod
    def unwrap_space(self, space):
        """
        Unwrap the space.

        Args:
            space: The space to unwrap.
        """
        pass

    @abstractmethod
    def wrap_point(self, space, point):
        """
        Wrap a point to the space.

        Args:
            space: The space into which to wrap the point.
            point: The point to wrap.
        """
        pass
    
    @abstractmethod
    def unwrap_point(self, space, point):
        """
        Unwrap a point to the space.

        Args:
            space: The space into which to unwrap the point.
            point: The point to unwrap.
        """
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
