
from gym.spaces import Dict, Discrete
import numpy as np

from admiral.envs.components.agent import PositionAgent, AgentObservingAgent

def obs_filter_step(distance, view):
    """
    Perfectly observe the agent if it is within the observing agent's view. If
    it is not within the view, then don't observe it at all.
    """
    return 0 if distance > view else 1

class PositionRestrictedObservationWrapper:
    """
    Partial observation based on position distance. If the observing agent is an
    AgentObservingAgent, then we will filter the observation based on the distance
    between the agent and the agent's view according to the given obs_filter function.
    We will also append that agent's observation with a "mask" channel that shows
    which agents have been observed and which have been filtered.

    We wrap multiple observers in one because you probably want to apply the same
    observation filter to many observers in the same step. For example, suppose
    your agent can observe the health and position of other agents. Supposed based
    on its position and view, another agent gets filtered out of the observation.
    We want that agent to be filtered out from both the position and health channels
    consistently, so we wrap both of those observers with a single wrapper.

    observers (list of Observers):
        All the observers to which you want to apply the same partial observation
        filter.
    
    obs_filter (function):
        A function with inputs distance and observing agent's view and outputs
        the probabilty of observing that agent.
        Default is obs_filter_step.
    
    obs_norm (int):
        The norm to use in calculating the distance.
        Default is np.inf.
    
    agents (dict):
        Dictionary of agents.
    """
    def __init__(self, observers, obs_filter=obs_filter_step, obs_norm=np.inf, agents=None, **kwargs):
        assert type(observers) is list, "observers must be in a list."
        self.observers = observers

        assert callable(obs_filter), "obs_filter must be a function."
        self.obs_filter = obs_filter

        assert type(obs_norm) is int, "obs_norm must be an integer or np.inf."
        self.obs_norm = obs_norm

        assert type(agents) is dict, "agents must be the dictionary of agents."
        self.agents = agents
        
        # Append a "mask" observation to the observing agents
        for agent in agents.values():
            if isinstance(agent, AgentObservingAgent):
                agent.observation_space['mask'] = Dict({
                    other: Discrete(2) for other in agents
                })

    def get_obs(self, agent, **kwargs):
        """
        Get the observation for this agent from the observers and filter based
        on the obs_filter.

        agent (ObservingAgent):
            An agent that can observe. If the agent does not have a position, then
            we cannot do position-based filtering, so we just return the observations
            wihtout a filter and with a mask that is all 1's for all agents. 
        
        return (dict):
            A dictionary composed of the channels from the observers and a "mask"
            channel that is 1 if the agent was observed, otherwise 0.
        """
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

            # Go through each observer and filter out the observations.
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
