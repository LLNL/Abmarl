
from abc import ABC, abstractmethod

# TODO: Component doesn't really make sense...
class Component(ABC):
    @abstractmethod
    def act(self, agent, *args, **kwargs):
        pass
