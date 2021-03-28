# Demoing an observation wrapper design for partial observations #

import numpy as np

from admiral.envs.components.observer import HealthObserver, LifeObserver, MaskObserver, \
    PositionObserver, RelativePositionObserver, SpeedObserver, AngleObserver, VelocityObserver, \
    TeamObserver
from admiral.envs.components.agent import PositionAgent

def obs_filter_step(distance, view):
    return 0 if distance > view else 1

class PositionRestrictedObservationWrapper:
    def __init__(self, observers, obs_filter=obs_filter_step, obs_norm=np.inf, agents=None, **kwargs):
        # TODO: Remove the mask observer and include masking here.
        self.observers = observers
        self.obs_filter = obs_filter
        self.agents = agents
        self.obs_norm = obs_norm

    def get_obs(self, agent, **kwargs):
        all_obs = {}
        if not isinstance(agent, PositionAgent):
            for observer in self.observers:
                all_obs.update(observer.get_obs(agent, **kwargs))
            return all_obs

        other_filter = []
        for other in self.agents.values():
            if other.id == agent.id: continue
            elif not isinstance(other, PositionAgent): continue # Cannot filter out agents who have no position
            elif np.random.uniform() <= \
                self.obs_filter(np.linalg.norm(agent.position - other.position, self.obs_norm), agent.agent_view): \
                    continue # We perfectly observed this agent
            else:
                other_filter.append(other.id)

        for observer in self.observers:
            obs = observer.get_obs(agent, **kwargs)
            for obs_type, obs_content in obs.items():
                for other in other_filter:
                    if other in obs_content: # TODO: observers should see all agents for consistent input, so this if check will go away when we fix that
                        obs_content[other] = observer.null_value

            all_obs.update(obs)
        
        return all_obs
