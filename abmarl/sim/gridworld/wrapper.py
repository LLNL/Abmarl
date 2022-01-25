
from abc import abstractmethod

from abmarl.sim.gridworld.actor import ActorBaseComponent
from abmarl.sim.gridworld.observer import ObserverBaseComponent
from abmarl.sim.gridworld.base import GridWorldBaseComponent
from abmarl.sim.wrappers import ravel_discrete_wrapper as rdw

class ComponentWrapper(GridWorldBaseComponent):
    """
    Wraps GridWorldBaseComponent.

    Every wrapper must be able to wrap and unwrap the respective space and points
    to and from that space. Agents and Grid are referenced directly from the wrapped
    component rather than received as initialization parameters.
    """
    @property
    @abstractmethod
    def wrapped_component(self):
        """
        Get the first-level wrapped component.
        """
        pass

    @property
    def unwrapped(self):
        """
        Fall through all the wrappers and obtain the original, completely unwrapped simulation.
        """
        try:
            return self.wrapped_component.unwrapped
        except AttributeError:
            return self.wrapped_component

    @property
    def agents(self):
        """
        The agent dictionary is directly taken from the wrapped component.
        """
        return self.wrapped_component.agents

    @property
    def grid(self):
        """
        The grid dictionary is directly taken from the wrapped component.
        """
        return self.wrapped_component.key

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
    def wrap_point(self, space, point):
        """
        Wrap a point to the space.

        Args:
            space: The space into which to wrap the point.
            point: The point to wrap.
        """
        pass

class ActorWrapper(ComponentWrapper, ActorBaseComponent):
    """
    Wraps an ActorComponent.

    Modify the action space of the agents involved with the Actor, namely the specific
    actor's channel. The actions recieved from the trainer are the wrapped space,
    so we need unwrap them to send them to the actor. This is the opposite from
    how we wrap and unwrap observations.
    """
    def __init__(self, component):
        assert isinstance(component, ActorBaseComponent), \
            "Wrapped component must be an ActorBaseComponent."
        self._actor = component
        # Need to record the pre-wrapped space for the wrapping functions.
        self.from_space = {
            agent.id: agent.action_space[self.key]
            for agent in self.agents.values()
            if isinstance(agent, self.supported_agent_type)
        }
        for agent in self.agents.values():
            if isinstance(agent, self.supported_agent_type):
                assert self.check_space(agent.action_space[self.key]), \
                    f"Cannot wrap {self.key} action channel for agent {agent.id}"
                agent.action_space[self.key] = self.wrap_space(agent.action_space[self.key])
        # TODO: Confirm that from_space is not just a reference to the space object
        # but is a unique copy of it, otherwise it would have been transformed
        # and is meaningless to have.

    @property
    def wrapped_component(self):
        """
        Get the wrapped actor.
        """
        return self._actor

    @property
    def key(self):
        """
        The key is the same as the wrapped actor's key.
        """
        return self._wrapped_component.key

    @property
    def supported_agent_type(self):
        """
        The supported agent type is the same as the wrapped actor's supported agent type.
        """
        return self._wrapped_component.supported_agent_type

    def process_action(self, agent, action_dict, **kwargs):
        """
        Unwrap the action and pass it to the wrapped actor to process.

        Args:
            agent: The acting agent.
            action_dict: The action dictionary for this agent in this step. The
                action in this channel comes in the wrapped space.
        """
        if isinstance(agent, self.supported_agent_type):
            action = action_dict[self.key]
            wrapped_action = self.wrap_point(self.from_space[agent.id], action)
            return self.wrapped_component.process_action(
                agent,
                {self.key: wrapped_action},
                **kwargs
            )

# TODO: Fill out the details of the observer wrapper.
class ObserverWrapper(ComponentWrapper, ObserverBaseComponent):
    pass


class RavelActionWrapper(ActorWrapper):
    def check_space(self, space):
        """
        Ensure that the space is of type that can be ravelled to discrete value.
        """
        return rdw.check_space(space)

    # TODO: Input should be agent, not space.
    def wrap_space(self, space):
        """
        Convert the space into a Discrete space.
        """
        return rdw.ravel_space(space)

    def wrap_point(self, space, point):
        """
        Unravel a single discrete point to a value in the space.

        Recall that the action from the trainer arrives in the wrapped discrete
        space, so we need to unravle it so that it is in the unwrapped space.
        """
        return rdw.unravel(space, point)

