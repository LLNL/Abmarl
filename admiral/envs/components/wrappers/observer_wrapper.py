# Demoing an observation wrapper design for partial observations #

import numpy as np

from admiral.envs.components.observer import HealthObserver, LifeObserver, MaskObserver, \
    PositionObserver, RelativePositionObserver, SpeedObserver, AngleObserver, VelocityObserver, \
    TeamObserver
from admiral.envs.components.agent import PositionAgent

def obs_filter_step(distance, view):
    return 0 if distance > view else 1

class PositionRestrictedObservationWrapper:
    def __init__(self, observer, obs_filter=obs_filter_step, obs_norm=np.inf, agents=None, **kwargs):
        self.observer = observer
        self.obs_filter = obs_filter
        self.agents = agents
        self.obs_norm = obs_norm
    
    def get_obs(self, agent, **kwargs):
        obs = self.observer.get_obs(agent, **kwargs)
        if not isinstance(agent, PositionAgent):
            # Agent does not have a position, so we cannot filter based on position.
            # We return the observation as is
            return obs
        else:
            for other in self.agents.values():
                if other.id == agent.id: continue
                elif not isinstance(other, PositionAgent): continue # Cannot filter out agents who have no position
                elif np.random.uniform() <= \
                    self.obs_filter(np.linalg.norm(agent.position - other.position, self.obs_norm), agent.agent_view): \
                        continue # We perfectly observed this agent
                else:
                    # We did not observe the agent, so we have to modify the obs
                    for obs_type, obs_content in obs.items():
                        if other.id in obs_content: # TODO: observer should see all agents for consistent input, so this if check will go away when we fix that
                            obs_content[other.id] = self.observer.null_value
        return obs
