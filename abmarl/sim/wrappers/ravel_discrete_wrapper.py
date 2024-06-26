import itertools

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, MultiBinary, Dict, Tuple
from gymnasium.spaces import Box as GymBox
from gymnasium.spaces.box import get_inf

from abmarl.sim import is_agent

from .sar_wrapper import SARWrapper


def _ravel_helper(space, point):
    if isinstance(space, Discrete):
        return point, space.n
    if isinstance(space, MultiDiscrete):
        return np.ravel_multi_index(point, space.nvec), np.prod(space.nvec)
    if isinstance(space, MultiBinary):
        return np.ravel_multi_index(point, [2] * space.n), 2 ** space.n
    if isinstance(space, GymBox):
        space_helper = (space.high + 1 - space.low).flatten()
        return np.ravel_multi_index((point - space.low).flatten(), space_helper), \
            np.prod(space_helper)
    elif isinstance(space, Dict):
        discretized_values = []
        space_dims = []
        for key, space in space.spaces.items():
            discretized_value, space_dim = _ravel_helper(space, point[key])
            discretized_values.append(discretized_value)
            space_dims.append(space_dim)
        return _ravel_helper(MultiDiscrete(space_dims), discretized_values)
    elif isinstance(space, Tuple):
        discretized_values = []
        space_dims = []
        for point_part, space in zip(point, space.spaces):
            discretized_value, space_dim = _ravel_helper(space, point_part)
            discretized_values.append(discretized_value)
            space_dims.append(space_dim)
        return _ravel_helper(MultiDiscrete(space_dims), discretized_values)


def _nested_dim_helper(space):
    if isinstance(space, Discrete):
        return [space.n]
    elif isinstance(space, MultiDiscrete):
        return [np.prod(space.nvec)]
    elif isinstance(space, MultiBinary):
        return [2 ** space.n]
    if isinstance(space, GymBox):
        return [np.prod(space.high + 1 - space.low)]
    elif isinstance(space, Dict):
        return [np.prod([_nested_dim_helper(s) for s in space.spaces.values()])]
    elif isinstance(space, Tuple):
        return [np.prod([_nested_dim_helper(s) for s in space.spaces])]
    else:
        raise TypeError


def _nested_dim(space):
    """
    Return the total number of dimensions in the entire (nested) space.
    """
    if isinstance(space, Dict):
        return [
            *itertools.chain.from_iterable([_nested_dim_helper(s) for s in space.spaces.values()])
        ]
    elif isinstance(space, Tuple):
        return [*itertools.chain.from_iterable([_nested_dim_helper(s) for s in space.spaces])]
    else:
        return _nested_dim_helper(space)


def ravel(space, point):
    """
    Ravel point in space to a single discrete value.
    """
    return _ravel_helper(space, point)[0]


def unravel(space, point):
    """
    Unravel a single discrete point to a value in the space.
    """
    if isinstance(space, Discrete):
        return point
    if isinstance(space, MultiDiscrete):
        return [*np.unravel_index(point, space.nvec)]
    if isinstance(space, MultiBinary):
        return [*np.unravel_index(point, [2] * space.n)]
    if isinstance(space, GymBox):
        space_helper = (space.high + 1 - space.low).flatten()
        return np.reshape(np.unravel_index(point, space_helper), space.shape) + space.low
    elif isinstance(space, Dict):
        dims = _nested_dim(space)
        unravelled_point = unravel(MultiDiscrete(dims), point)
        output = {}
        for i, (key, value) in enumerate(space.spaces.items()):
            output[key] = unravel(value, unravelled_point[i])
        return output
    elif isinstance(space, Tuple):
        dims = _nested_dim(space)
        unravelled_point = unravel(MultiDiscrete(dims), point)
        output = []
        for i, value in enumerate(space.spaces):
            output.append(unravel(value, unravelled_point[i]))
        return tuple(output)


def ravel_space(space):
    """
    Convert the space into a Discrete space.
    """
    dims = _nested_dim_helper(space)
    return Discrete(dims[0])


def _isbounded(space):
    """
    Gym Box converts np.inf to min and max values for integer types. As a result,
    Box.is_bounded doesn't work because it checks for inf, not for min/max values
    of that dtype. This function checks for min/max values of the dtype.
    """
    return space.is_bounded() and \
        not (space.low == get_inf(int, '-')).any() and \
        not (space.low == get_inf(int, '+')).any() and \
        not (space.high == get_inf(int, '-')).any() and \
        not (space.high == get_inf(int, '+')).any()


def check_space(space):
    """
    Ensure that the space is of type that can be ravelled to discrete value.
    """
    if isinstance(space, Discrete) or isinstance(space, MultiDiscrete) or \
            isinstance(space, MultiBinary):
        return True
    elif isinstance(space, GymBox) and np.issubdtype(space, int) and _isbounded(space):
        return True
    elif isinstance(space, Dict):
        return all([check_space(sub_space) for sub_space in space.spaces.values()])
    elif isinstance(space, Tuple):
        return all([check_space(sub_space) for sub_space in space.spaces])
    else:
        return False


class RavelDiscreteWrapper(SARWrapper):
    """
    Convert observation and action spaces into a Discrete space.

    Convert Discrete, MultiBinary, MultiDiscrete, bounded integer Box, and any nesting of these
    observations and actions into Discrete observations and actions by "ravelling" their values
    according to numpy's ravel_mult_index function. Thus, observations and actions that are
    represented by arrays are converted into unique numbers. This is useful for building Q
    tables where each observation and action is a row and column of the Q table, respectively.

    If the agent has a null observation or a null action, that value is also ravelled
    into the new Discrete space.
    """
    def __init__(self, sim):
        super().__init__(sim)
        for agent_id, wrapped_agent in self.agents.items():
            if not is_agent(wrapped_agent): continue
            assert check_space(wrapped_agent.observation_space), \
                f"{agent_id}: observation must be discretizable."
            assert check_space(wrapped_agent.action_space), \
                f"{agent_id} action must be discretizable."
            self.agents[agent_id].observation_space = ravel_space(wrapped_agent.observation_space)
            self.agents[agent_id].action_space = ravel_space(wrapped_agent.action_space)
            if self.agents[agent_id].null_observation:
                self.agents[agent_id].null_observation = ravel(
                    self.sim.agents[agent_id].observation_space,
                    wrapped_agent.null_observation
                )
            if self.agents[agent_id].null_action:
                self.agents[agent_id].null_action = ravel(
                    self.sim.agents[agent_id].action_space,
                    wrapped_agent.null_action
                )

    def wrap_observation(self, from_agent, observation):
        return ravel(from_agent.observation_space, observation)

    def unwrap_observation(self, from_agent, observation):
        return unravel(from_agent.observation_space, observation)

    def wrap_action(self, from_agent, action):
        return unravel(from_agent.action_space, action)

    def unwrap_action(self, from_agent, action):
        return ravel(from_agent.action_space, action)
