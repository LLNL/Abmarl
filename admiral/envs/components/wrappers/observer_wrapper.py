# Demoing an observation wrapper design for partial observations #

from gym.spaces import Dict, Discrete
import numpy as np

from admiral.envs.components.observer import HealthObserver, LifeObserver, MaskObserver, \
    PositionObserver, RelativePositionObserver, SpeedObserver, AngleObserver, VelocityObserver, \
    TeamObserver, AgentObservingAgent
from admiral.envs.components.agent import PositionAgent

def obs_filter_step(distance, view):
    return 0 if distance > view else 1

class PositionRestrictedObservationWrapper:
    def __init__(self, observers, obs_filter=obs_filter_step, obs_norm=np.inf, agents=None, **kwargs):
        self.observers = observers
        self.obs_filter = obs_filter
        self.obs_norm = obs_norm
        self.agents = agents
        
        # Append a "mask" observation to the observing agents
        for agent in agents.values():
            if isinstance(agent, AgentObservingAgent):
                agent.observation_space['mask'] = Dict({
                    other: Discrete(2) for other in agents
                })

    def get_obs(self, agent, **kwargs):
        if isinstance(agent, AgentObservingAgent):
            all_obs = {}

            # If the observing agent does not have a position, then we cannot filter
            # it here, so we just return the observations from the wrapped observers.
            if not isinstance(agent, PositionAgent):
                mask = {other: 1 for other in self.agents}
                all_obs['mask'] = mask
                for observer in self.observers:
                    all_obs.update(observer.get_obs(agent, **kwargs))
                return all_obs

            # Determine which other agents the observing agent sees. Add the observation mask.
            mask = {}
            for other in self.agents.values():
                if not isinstance(other, PositionAgent) or \
                    np.random.uniform() <= self.obs_filter(np.linalg.norm(agent.position - other.position, self.obs_norm), agent.agent_view):
                    mask[other.id] = 1 # We perfectly observed this agent
                else:
                    mask[other.id] = 0 # We did not observe this agent
            all_obs['mask'] = mask

            for observer in self.observers:
                obs = observer.get_obs(agent, **kwargs)
                for obs_content in obs.values():
                    for other, masked in mask.items():
                        if not masked:
                            obs_content[other] = observer.null_value

                all_obs.update(obs)
            
            return all_obs
        else:
            return {}
