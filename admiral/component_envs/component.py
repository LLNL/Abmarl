
from abc import ABC, abstractmethod

class Component(ABC):
    @abstractmethod
    def act(self, agent, *args, **kwargs):
        pass
