from gym.spaces import Space, Discrete, MultiBinary, MultiDiscrete, Box, Dict, Tuple


def check_space(space, strict=False):
    """
    Ensure that the space is a gym Space, including all nested spaces.

    strict (bool), default False:
        If strict is True, then the recursion rule is that every subspace must be
        a gym space. If strict is False, then the recursion rule is that every subspace
        must be a gym space OR a dict or tuple. In this way, we allow the space
        to be iteratively built and assume that the final wrapping to Dict or Tuple
        has yet to occur.
    """
    if isinstance(space, (Discrete, MultiDiscrete, MultiBinary, Box)):
        return True
    elif isinstance(space, Dict):
        return all([check_space(sub_space) for sub_space in space.spaces.values()])
    elif isinstance(space, Tuple):
        return all([check_space(sub_space) for sub_space in space.spaces])
    elif not strict:
        if isinstance(space, dict):
            return all([check_space(sub_space) for sub_space in space.values()])
        elif isinstance(space, tuple):
            return all([check_space(sub_space) for sub_space in space])
    else:
        return False


def make_dict(space):
    """
    Convert a hierarchical space into a gym space by recursively moving through
    the layers and converting the subspaces to gym spaces. Unsafe, modifies the
    items of the input as it moves through them.
    """
    assert isinstance(space, (dict, Space)), "Cannot convert this to a Dict."
    for key, subspace in space.items():
        if isinstance(subspace, dict):
            space[key] = make_dict(subspace)
        else:
            assert isinstance(subspace, Space), "Cannot convert this to a Dict."

    return Dict(space) if type(space) is dict else space
