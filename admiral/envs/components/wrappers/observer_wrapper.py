
from gym.spaces import Dict, Discrete
import numpy as np

from admiral.envs.components.agent import PositionAgent, AgentObservingAgent, ObservingAgent, BroadcastingAgent, TeamAgent

def obs_filter_step(distance, view):
    """
    Perfectly observe the agent if it is within the observing agent's view. If
    it is not within the view, then don't observe it at all.
    """
    return 0 if distance > view else 1

class PositionRestrictedObservationWrapper:
    """
    Partial observation based on position distance. If the observing agent is an
    AgentObservingAgent, then we will filter the observation based on agent's view
    and the distance between the agents according to the given obs_filter function.
    We will also append that agent's observation with a "mask" channel that shows
    which agents have been observed and which have been filtered.

    We wrap multiple observers in one because you probably want to apply the same
    observation filter to many observers in the same step. For example, suppose
    your agent can observe the health and position of other agents. Suppose based
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
        self._channel_observer_map = {observer.channel: observer for observer in self.observers}

        assert callable(obs_filter), "obs_filter must be a function."
        self.obs_filter = obs_filter

        self.obs_norm = obs_norm

        assert type(agents) is dict, "agents must be the dictionary of agents."
        self.agents = agents
        
        # Append a "mask" observation to the observing agents
        for agent in agents.values():
            if isinstance(agent, ObservingAgent):
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
            without a filter and with a mask that is all 1's for all agents. 
        
        return (dict):
            A dictionary composed of the channels from the observers and a "mask"
            channel that is 1 if the agent was observed, otherwise 0.
        """
        if isinstance(agent, ObservingAgent):
            all_obs = {}

            # If the observing agent does not have a position and view, then we cannot filter
            # it here, so we just return the observations from the wrapped observers.
            if not (isinstance(agent, PositionAgent) and isinstance(agent, AgentObservingAgent)):
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
    
    def null_value(self, channel):
        if channel == 'mask':
            return 0
        else:
            return self._channel_observer_map[channel].null_value

class TeamBasedCommunicationWrapper:
    """
    Agents can broadcast their observations for other agents to see. Other agents
    that are on the same team will receive the observation, and it will be fused
    in with their own observations. Other agents that are on a different team will
    not receive the message contents but will receive information about the broadcasting
    agent, such as its location.

    Note: This wrapper only works with channel observers that are keyed off the
    agents' ids.

    observers (list of PositionRestrictedObservationWrapper):
        The PositionRestrictedObservationWrapper masks agents from one another,
        making it necessary for communications to occur in order to reveal observations.
    
    agents (dict):
        The dictionary of agents.
    
    obs_norm (int):
        The norm to use for measuring the distance between agents.
        Default np.inf.
    """
    def __init__(self, observers, agents=None, obs_norm=np.inf, **kwargs):
        self.observers = observers
        self.agents = agents
        self.obs_norm = obs_norm

    def get_obs(self, receiving_agent, **kwargs):
        """
        If the receiving is within range of a broadcasting agent, it will receive
        a message. If they're on the same team, then the receiving agent will augment
        its observation with the observation of the broadcasting agent. If they
        are not different teams, then the receiving agent will not receive the
        observation of the broadcasting agent but will receive information about
        that agent itself.
        """
        if isinstance(receiving_agent, ObservingAgent):
            # Generate my normal observation
            my_obs = {}
            for observer in self.observers:
                my_obs.update(observer.get_obs(receiving_agent, **kwargs))

            # Fuse my observation with information from the broadcasting agent.
            # If I'm on the same team, then I will see its observation.
            # If I'm not on the same team, then I will not see its observation,
            #   but I will still see its own attributes.
            for broadcasting_agent in self.agents.values():
                if isinstance(broadcasting_agent, PositionAgent) and isinstance(receiving_agent, PositionAgent) and broadcasting_agent.broadcasting:
                    distance = np.linalg.norm(broadcasting_agent.position - receiving_agent.position, self.obs_norm)
                    if distance > broadcasting_agent.broadcast_range: continue # Too far from this broadcasting agent
                    elif isinstance(receiving_agent, TeamAgent) and isinstance(broadcasting_agent, TeamAgent) and receiving_agent.team == broadcasting_agent.team:
                        # Broadcasting and receiving agent are on the same team, so the receiving agent receives the observation
                        # broadcasting_obs ={}
                        for observer in self.observers:
                            tmp_obs = observer.get_obs(broadcasting_agent, **kwargs)
                            for obs_type, obs_content in tmp_obs.items():
                                for agent_id, obs_value in obs_content.items():
                                    if np.all(my_obs[obs_type][agent_id] == observer.null_value(obs_type)):
                                        my_obs[obs_type][agent_id] = obs_value
                    else:
                        # I received a message, but we're not on the same team, so I only observe the
                        # broadcasting agent's information. Since I don't have a state, I have to go
                        # through the agent's observation of itself.
                        # NOTE: This is a bit of a hack. It's not fullproof because the broadcasting
                        # agent might not have information about itself. This is the best we can do
                        # right now without a re-design.
                        for observer in self.observers:
                            tmp_obs = observer.get_obs(broadcasting_agent, **kwargs)
                            for obs_type, obs_content in tmp_obs.items():
                                if np.all(my_obs[obs_type][broadcasting_agent.id] == observer.null_value(obs_type)):
                                    my_obs[obs_type][broadcasting_agent.id] = obs_content[broadcasting_agent.id]
            return my_obs
        else:
            return {}
                



