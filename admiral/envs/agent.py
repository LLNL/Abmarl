
class Agent:
    """
    Base Agent class for agents that live in an environment. Agents require an
    id in in order to even be constructed. Agents must also have an observation
    space and action space to be considered successfully configured.
    """
    def __init__(self, id=None, observation_space={}, action_space={}, **kwargs):
        if id is None:
            raise TypeError("Agents must be constructed with an id.")
        else:
            self.id = id
        self.observation_space = observation_space
        self.action_space = action_space
    
    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return self.id is not None and self.action_space and self.observation_space

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ if isinstance(other, self.__class__) else False
