
from gym.spaces import Box, Discrete, Tuple, Dict, MultiDiscrete, MultiBinary
import numpy as np

from abmarl.sim import Agent

from .sar_wrapper import SARWrapper


def flatdim(space):
    """
    Return the number of dimensions a flattened equivalent of this space
    would have.

    Args:
        space: A gym Space.

    Returns:
        The number of dimensions in the flattened space.
    """
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return 1
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return len(space)
    elif isinstance(space, Tuple):
        return int(sum([flatdim(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([flatdim(s) for s in space.spaces.values()]))


def flatten(space, point):
    """
    Flatten a point from a space.

    This is useful when e.g. points from spaces must be passed to a neural
    network, which only understands flat arrays of floats.

    Args:
        space: The gym space in which the point lives
        point: The point to be flattened.

    Returns:
        A flattened representation of the point.
    """
    if isinstance(space, Box):
        return np.asarray(point, dtype=space.dtype).flatten()
    elif isinstance(space, Discrete):
        return np.array([point], dtype=int)
    elif isinstance(space, Tuple):
        return np.concatenate([flatten(s, x_part) for x_part, s in zip(point, space.spaces)])
    elif isinstance(space, Dict):
        return np.concatenate(
            [flatten(s, point[key]) for key, s in space.spaces.items()])
    elif isinstance(space, MultiBinary):
        return point
    elif isinstance(space, MultiDiscrete):
        return point


def unflatten(space, point):
    """
    Unflatten a point to a space.

    This reverses the transformation applied by flatten(). You must ensure
    that the space argument is the same as for the flatten() call.

    Args:
        space: The gym space to which to map the point.
        point: The point to be unflattened.

    Returns:
        An unflattened representation of the point.
    """
    if isinstance(space, Box):
        return np.asarray(point, dtype=space.dtype).reshape(space.shape)
    elif isinstance(space, Discrete):
        return point[0]
    elif isinstance(space, MultiBinary):
        return point
    elif isinstance(space, MultiDiscrete):
        return point
    elif isinstance(space, Tuple):
        dims = [flatdim(s) for s in space.spaces]
        list_flattened = np.split(point, np.cumsum(dims)[:-1])
        list_unflattened = [
            unflatten(s, flattened)
            for flattened, s in zip(list_flattened, space.spaces)
        ]
        return tuple(list_unflattened)
    elif isinstance(space, Dict):
        dims = [flatdim(s) for s in space.spaces.values()]
        list_flattened = np.split(point, np.cumsum(dims)[:-1])
        list_unflattened = [
            (key, unflatten(s, flattened))
            for flattened, (key,
                            s) in zip(list_flattened, space.spaces.items())
        ]
        from collections import OrderedDict
        return OrderedDict(list_unflattened)


def flatten_space(space):
    """Flatten a space into a single Box.

    This is equivalent to flatten(), but operates on the space itself. The
    result always is a Box with flat boundaries. The box has exactly
    flatdim(space) dimensions. Flattening a sample of the original space
    has the same effect as taking a sample of the flattenend space.

    Args:
        space: Gym space to be flattened.

    Returns:
        Box with type and dimension according to the flattening.
    """
    if isinstance(space, Box):
        return Box(space.low.flatten(), space.high.flatten(), dtype=space.dtype)
    if isinstance(space, Discrete):
        return Box(low=0, high=space.n - 1, shape=(1, ), dtype=int)
    if isinstance(space, MultiBinary):
        return Box(low=0, high=1, shape=(space.n, ), dtype=int)
    if isinstance(space, MultiDiscrete):
        return Box(
            low=np.zeros_like(space.nvec),
            high=space.nvec - 1,
            dtype=int
        )
    if isinstance(space, Tuple):
        space = [flatten_space(s) for s in space.spaces]
        encapsulating_type = int \
            if all([this_space.dtype == int for this_space in space]) \
            else float
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
            dtype=encapsulating_type
        )
    if isinstance(space, Dict):
        space = [flatten_space(s) for s in space.spaces.values()]
        encapsulating_type = int \
            if all([this_space.dtype == int for this_space in space]) \
            else float
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
            dtype=encapsulating_type
        )


class FlattenWrapper(SARWrapper):
    """
    Flattens all agents' action and observation spaces into Boxes.

    Nested spaces (e.g. Tuples and Dicts) are flattened element-wise, each element
    being concatentated onto the previous. A Discrete space is converted to a Box
    with a single element, whose bounds are 0 to ``space.n`` - 1. MultiBinary and MultiDiscrete
    are simply converted to Box with the corresponding bounds and integer dtype.
    A Box space is flattened to a one-dimensional array equivalent.

    If the resulting Box can be made with dtype int, then it will be. Otherwise,
    it will cast up to float.

    If the agent has a null observation or a null action, that value is also flattened
    into the new Box space.

    NOTE: Sampling from the flattened space will not produce the same results as
    sampling from the original space and then flattening.
    """
    def __init__(self, sim):
        super().__init__(sim)
        for agent_id, wrapped_agent in self.sim.agents.items(): # Wrap the agents' spaces
            if not isinstance(wrapped_agent, Agent): continue
            self.agents[agent_id].action_space = flatten_space(wrapped_agent.action_space)
            self.agents[agent_id].observation_space = flatten_space(
                wrapped_agent.observation_space
            )
            if self.agents[agent_id].null_observation:
                self.agents[agent_id].null_observation = flatten(
                    self.sim.agents[agent_id].observation_space,
                    wrapped_agent.null_observation
                )
            if self.agents[agent_id].null_action:
                self.agents[agent_id].null_action = flatten(
                    self.sim.agents[agent_id].action_space,
                    wrapped_agent.null_action
                )

    def wrap_observation(self, from_agent, observation):
        return flatten(from_agent.observation_space, observation)

    def unwrap_observation(self, from_agent, observation):
        return unflatten(from_agent.observation_space, observation)

    def wrap_action(self, from_agent, action):
        return unflatten(from_agent.action_space, action)

    def unwrap_action(self, from_agent, action):
        return flatten(from_agent.action_space, action)


class FlattenActionWrapper(SARWrapper):
    """
    Flattens all agents' action spaces into continuous Boxes.
    """
    def __init__(self, sim):
        super().__init__(sim)
        for agent_id, wrapped_agent in self.sim.agents.items():
            if not isinstance(wrapped_agent, Agent): continue
            # Wrap the action spaces of the agents
            self.agents[agent_id].action_space = flatten_space(wrapped_agent.action_space)
            if self.agents[agent_id].null_action:
                self.agents[agent_id].null_action = flatten(
                    self.sim.agents[agent_id].action_space,
                    wrapped_agent.null_action
                )

    def wrap_action(self, from_agent, action):
        return unflatten(from_agent.action_space, action)

    def unwrap_action(self, from_agent, action):
        return flatten(from_agent.action_space, action)
