
from gym.spaces import Dict, MultiBinary

from admiral.envs import Agent
from admiral.envs.components.agent import AgentObservingAgent
from admiral.envs.components.position import PositionAgent, PositionObserver, RelativePositionObserver
from admiral.envs.components.team import TeamObserver
from admiral.envs.components.health import HealthObserver, LifeObserver

from admiral.envs.components.observer import MaskObserver, PositionRestrictedMaskObserver, \
    PositionRestrictedTeamObserver, PositionRestrictedPositionObserver, \
    PositionRestrictedRelativePositionObserver, PositionRestrictedHealthObserver, \
    PositionRestrictedLifeObserver

