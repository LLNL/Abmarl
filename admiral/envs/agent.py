
def Agent(id=None, observation_space=None, action_space=None, **kwargs):
    assert id is not None, "id must not be None"
    return {
        'id': id,
        'observation_space': {} if observation_space is None else observation_space,
        'action_space': {} if action_space is None else action_space,
    }
