
from abc import abstractmethod

from gym.spaces import Discrete, Dict

from abmarl.sim.gridworld.actor import ActorBaseComponent
from abmarl.sim.gridworld.observer import ObserverBaseComponent
from abmarl.sim.gridworld.base import GridWorldBaseComponent
from abmarl.sim.wrappers import ravel_discrete_wrapper as rdw


class ComponentWrapper(GridWorldBaseComponent):
    """
    Wraps GridWorldBaseComponent.

    Every wrapper must be able to wrap the respective space and points
    to/from that space. Agents and Grid are referenced directly from the wrapped
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
        Fall through all the wrappers and obtain the original, completely unwrapped component.
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
        The grid is directly taken from the wrapped component.
        """
        return self.wrapped_component.grid

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
        Wrap a point using a reference space.

        Args:
            space: The reference space for wrapping the point.
            point: The point to wrap.
        """
        pass

    @abstractmethod
    def unwrap_point(self, space, point):
        """
        Unwrap a point using a reference space.

        Args:
            space: The reference space for unwrapping the point.
            point: The point to unwrap.
        """
        pass


class ActorWrapper(ComponentWrapper, ActorBaseComponent):
    """
    Wraps an ActorComponent.

    Modify the action space of the agents involved with the Actor, namely the specific
    actor's channel. The actions recieved from the trainer are in the wrapped space,
    so we need to unwrap them to send them to the actor. This is the opposite from
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
                if agent.null_action:
                    agent.null_action[self.key] = self.unwrap_point(
                        self.from_space[agent.id],
                        agent.null_action[self.key]
                    )

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
        return self.wrapped_component.key

    @property
    def supported_agent_type(self):
        """
        The supported agent type is the same as the wrapped actor's supported agent type.
        """
        return self.wrapped_component.supported_agent_type

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
            unwrapped_action = self.wrap_point(self.from_space[agent.id], action)
            return self.wrapped_component.process_action(
                agent,
                {self.key: unwrapped_action},
                **kwargs
            )


# TODO Abmarl-202: Fill out the details of the abstract observer wrapper.
class ObserverWrapper(ComponentWrapper, ObserverBaseComponent):
    pass


# Docs for ObserverWrapper:
"""
Observer Wrappers
~~~~~~~~~~~~~~~~~

An Observer Wrapper uses the ``get_obs`` function, at which point
it can request an observation by passing the request to the underlying Observer
and then modify the data from the observer before sending it out. Observer Wrappers
may need to modifiy the observation spaces of corresponding agents to ensure that
the Trainer is expecting the correct format.
"""


class RavelActionWrapper(ActorWrapper):
    """
    Use numpy's ravel capabilities to convert space and points to Discrete.
    """
    def check_space(self, space):
        """
        Ensure that the space is of type that can be ravelled to discrete value.
        """
        return rdw.check_space(space)

    def wrap_space(self, space):
        """
        Convert the space into a Discrete space.
        """
        return rdw.ravel_space(space)

    def wrap_point(self, space, point):
        """
        Unravel a single discrete point to a value in the space.

        Recall that the action from the trainer arrives in the wrapped discrete
        space, so we need to unravel it so that it is in the unwrapped space before
        giving it to the actor.
        """
        return rdw.unravel(space, point)

    def unwrap_point(self, space, point):
        """
        Ravel point to a single discrete value.
        """
        return rdw.ravel(space, point)


class ExclusiveChannelActionWrapper(ActorWrapper):
    """
    Ravel Dict space and points with top-level exclusion.

    This wrapper works with Dict spaces, where each subspace is to be ravelled
    independently and then combined so that that actions are exclusive. The wrapping
    occurs in two steps. First, we use numpy's ravel capabilities to convert each
    subspace to a Discrete space. Second, we combine the Discrete spaces together
    in such a way that imposes exclusivity among the subspaces. The exclusion happens
    only on the top level, so a Dict nested within a Dict will be ravelled without
    exclusion.
    """
    def check_space(self, space):
        """
        Top level must be Dict and subspaces must be ravel-able.
        """
        assert isinstance(space, Dict), "ExclusiveRavelActionWrapper works on Dict spaces."
        return rdw.check_space(space)

    def wrap_space(self, space):
        """
        Convert the space into a Discrete space.

        The wrapping occurs in two steps. First, we use numpy's ravel capabilities
        to convert each subspace to a Discrete space. Second, we combine the Discrete
        spaces together, imposing that actions among the subspaces are exclusive.
        """
        exclusive_channels = {
            channel: rdw.ravel_space(subspace) for channel, subspace in space.spaces.items()
        }
        # Using sum is the difference between exclusive and non-exclusive.
        # The rule is: sum for exclusive, prod for non-exclusive. The ravel function
        # uses prod by default, so we implement sum here directly to impose exclusivity.
        dims = sum([subspace.n for subspace in exclusive_channels.values()])
        # We modify the wrapping process to suppress duplicate zero vectors
        dims = dims - len(exclusive_channels) + 1
        return Discrete(dims)

    def wrap_point(self, space, point):
        """
        Unravel a single discrete point to a value in the space.

        Recall that the action from the trainer arrives in the wrapped discrete
        space, so we need to unravel it so that it is in the unwrapped space before
        giving it to the actor.
        """
        # Find the activated channel
        exclusive_channels = {
            channel: rdw.ravel_space(subspace) for channel, subspace in space.spaces.items()
        }

        for activated_channel, subspace in exclusive_channels.items():
            if point < subspace.n:
                break
            else:
                point = point - subspace.n + 1 # Add one to avoid duplicate zero vector

        # Unravel the point for the activated channel. The other channels unravel 0.
        output = {}
        for channel, subspace in space.items():
            if channel == activated_channel:
                output[channel] = rdw.unravel(subspace, point)
            else:
                output[channel] = rdw.unravel(subspace, 0)

        return output

    def unwrap_point(self, space, point):
        """
        Ravel point to a single discrete value.
        """
        exclusive_channels = {
            channel: rdw.ravel_space(subspace) for channel, subspace in space.spaces.items()
        }

        top_level_ravel = {
            channel: rdw.ravel(subspace, point[channel]) for channel, subspace in space.items()
        }

        ravelled_point = 0
        for channel, top_level_point in top_level_ravel.items():
            if top_level_point != 0:
                ravelled_point = ravelled_point + top_level_point
                break
            else:
                ravelled_point = ravelled_point + exclusive_channels[channel].n - 1
        # If the top level point is zero when we exit this loop, that means that
        # every value in the dict was 0, so we should return 0. If the top level
        # point is not zero, then we found a nonzero channel, so we should return
        # the ravelled point.
        return 0 if top_level_point == 0 else ravelled_point
