
from gym.spaces import Discrete, Dict

from abmarl.sim.wrappers import Wrapper
from abmarl.sim.wrappers import ravel_discrete_wrapper as rdw

class ExclusiveChannelActionWrapper(Wrapper):
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
    def __init__(self, sim):
        super().__init__(sim)
        for agent_id, wrapped_agent in self.agents.items():
            if not isinstance(wrapped_agent, Agent): continue
            assert self.check_space(wrapped_agent.action_space), \
                f"{agent_id} action cannot be made exclusive."
            self.agents[agent_id].action_space = self.wrap_space(wrapped_agent.action_space)
            if wrapped_agent:
                self.agents[agent_id].null_action = self.wrap_point(
                    self.sim.agents[agent_id].action_space,
                    wrapped_agent.null_action
                )

    def step(self, action_dict, **kwargs):
        """
        Wrap each of the agent's actions from the policies before passing them
        to sim.step.
        """
        self.sim.step(
            {
                agent_id: self.wrap_point(self.sim.agents[agent_id], action)
                for agent_id, action in action_dict.items()
            },
            **kwargs
        )

    def check_space(self, space):
        """
        Top level must be Dict and subspaces must be ravel-able.
        """
        assert isinstance(space, Dict), "ExclusiveChannelActionWrapper works on Dict spaces."
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
