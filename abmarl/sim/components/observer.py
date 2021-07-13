from abc import ABC, abstractmethod, abstractproperty

from gym.spaces import Box, Dict
import numpy as np

from abmarl.sim.components.agent import HealthObservingAgent, LifeObservingAgent, \
    AgentObservingAgent, PositionObservingAgent, SpeedAngleObservingAgent, \
    VelocityObservingAgent, ResourceObservingAgent, TeamObservingAgent, BroadcastObservingAgent, \
    SpeedAngleAgent, VelocityAgent, BroadcastingAgent, ComponentAgent


class Observer(ABC):
    """
    Base observer class provides the interface required of all observers. Setup
    the agents' observation space according to the Observer's channel.

        agents (dict):
            The dictionary of agents.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents

    def _set_obs_space_simple(self, instance, space_func, **kwargs):
        """
        Observers that don't depend on the type of the other agents can use this
        method.

        instance (Agent):
            An Agent class. This is used in the isinstance check to determine if
            the agent will receive the observation channel.

        space_func (function):
            A function that takes the other agent as input and outputs the
            observation space.
        """
        for agent in self.agents.values():
            if isinstance(agent, instance):
                agent.observation_space[self.channel] = Dict({
                    other.id: space_func(other) for other in self.agents.values()
                })

    def _set_obs_space(self, instance, other_instance, space_func, alt_space_func, **kwargs):
        """
        Observers that depend on the type of the other agents must use this method.

        instance (Agent):
            An Agent class. This is used in the isinstance check to determine if
            the agent will receive the observation channel.

        other_instance (Agent):
            An Agent class. This is used in the isinstance check on the other agents
            to determine whether to use the space_func or the alt_space_func.

        space_func (function):
            A function that takes the other agent as input and outputs the
            observation space.

        alt_space_func (function):
            Use this function for cases when the isinstance check fails on the
            other agent. Function does not have inputs and outputs observation space.
        """
        for agent in self.agents.values():
            if isinstance(agent, instance):
                obs_space = {}
                for other in self.agents.values():
                    if isinstance(other, other_instance):
                        obs_space[other.id] = space_func(other)
                    else:
                        obs_space[other.id] = alt_space_func()
                agent.observation_space[self.channel] = Dict(obs_space)

    def _get_obs(self, agent, instance=None, other_instance=ComponentAgent, attr=None, **kwargs):
        """
        Many observers just directly query the corresponding state field from the
        agent. This function does exactly that, checking the instance of the observing
        agent and the other agents and setting the observation value accordingly.
        """
        if isinstance(agent, instance):
            obs = {}
            for other in self.agents.values():
                if isinstance(other, other_instance):
                    obs[other.id] = getattr(other, attr)
                else:
                    obs[other.id] = self.null_value
            return {self.channel: obs}
        else:
            return {}

    @abstractmethod
    def get_obs(self, agent, **kwargs):
        pass

    @abstractproperty
    def channel(self):
        pass

    @abstractproperty
    def null_value(self):
        pass


# --------------------- #
# --- Communication --- #
# --------------------- #

class BroadcastObserver(Observer):
    """
    Observe the broadcast state of broadcasting agents.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_obs_space_simple(
            BroadcastObservingAgent, lambda *args: Box(-1, 1, (1,)), **kwargs
        )

    def get_obs(self, agent, **kwargs):
        return self._get_obs(
            agent,
            instance=BroadcastObservingAgent,
            other_instance=BroadcastingAgent,
            attr='broadcasting',
            **kwargs
        )

    @property
    def channel(self):
        return 'broadcast'

    @property
    def null_value(self):
        return -1


# ----------------------- #
# --- Health and Life --- #
# ----------------------- #

class HealthObserver(Observer):
    """
    Observe the health state of all the agents in the simulator.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_obs_space_simple(
            HealthObservingAgent, lambda other: Box(-1, other.max_health, (1,)), **kwargs
        )

    def get_obs(self, agent, **kwargs):
        """
        Get the health state of all the agents in the simulator.

        agent (HealthObservingAgent):
            The agent making the observation.
        """
        return self._get_obs(
            agent,
            instance=HealthObservingAgent,
            attr='health',
            **kwargs
        )

    @property
    def channel(self):
        return 'health'

    @property
    def null_value(self):
        return -1


class LifeObserver(Observer):
    """
    Observe the life state of all the agents in the simulator.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_obs_space_simple(
            LifeObservingAgent, lambda *args: Box(-1, 1, (1,), np.int), **kwargs
        )

    def get_obs(self, agent, **kwargs):
        """
        Get the life state of all the agents in the simulator.

        agent (LifeObservingAgent):
            The agent making the observation.
        """
        return self._get_obs(
            agent,
            instance=LifeObservingAgent,
            attr='is_alive',
            **kwargs
        )

    @property
    def channel(self):
        return 'life'

    @property
    def null_value(self):
        return -1


# ----------------------------- #
# --- Position and Movement --- #
# ----------------------------- #

class PositionObserver(Observer):
    """
    Observe the positions of all the agents in the simulator.
    """
    def __init__(self, position_state=None, **kwargs):
        super().__init__(**kwargs)
        self.position_state = position_state
        self._set_obs_space_simple(
            PositionObservingAgent,
            lambda *args: Box(-1, self.position_state.region, (2,), np.int), **kwargs
        )

    def get_obs(self, agent, **kwargs):
        """
        Get the positions of all the agents in the simulator.
        """
        return self._get_obs(agent, instance=PositionObservingAgent, attr='position')

    @property
    def channel(self):
        return 'position'

    @property
    def null_value(self):
        return np.array([-1, -1])


class RelativePositionObserver(Observer):
    """
    Observe the relative positions of agents in the simulator.
    """
    def __init__(self, position_state=None, **kwargs):
        super().__init__(**kwargs)
        self.position_state = position_state
        self._set_obs_space_simple(
            PositionObservingAgent,
            lambda *args: Box(
                -self.position_state.region, self.position_state.region, (2,), np.int
            ), **kwargs
        )

    def get_obs(self, agent, **kwargs):
        """
        Get the relative positions of all the agents in the simulator.
        """
        if isinstance(agent, PositionObservingAgent):
            obs = {}
            for other in self.agents.values():
                r_diff = other.position[0] - agent.position[0]
                c_diff = other.position[1] - agent.position[1]
                obs[other.id] = np.array([r_diff, c_diff])
            return {self.channel: obs}
        else:
            return {}

    @property
    def channel(self):
        return 'relative_position'

    @property
    def null_value(self):
        return np.array([-self.position_state.region, -self.position_state.region])


class GridPositionBasedObserver:
    """
    Agents observe a grid of size agent_view centered on their
    position. The values of the cells are as such:
        Out of bounds  : -1
        Empty          :  0
        Agent occupied : 1

    position (PositionState):
        The position state handler, which contains the region.

    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position_state=None, agents=None, **kwargs):
        self.position_state = position_state
        self.agents = agents

        for agent in agents.values():
            if isinstance(agent, AgentObservingAgent) and \
               isinstance(agent, PositionObservingAgent):
                agent.observation_space['position'] = Box(
                    -1, 1, (agent.agent_view*2+1, agent.agent_view*2+1), np.int
                )

    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position.
        """
        if isinstance(my_agent, AgentObservingAgent) and \
           isinstance(my_agent, PositionObservingAgent):
            signal = np.zeros((my_agent.agent_view*2+1, my_agent.agent_view*2+1))

            # --- Determine the boundaries of the agents' grids --- #
            # For left and top, we just do: view - x,y >= 0
            # For the right and bottom, we just do region - x,y - 1 - view > 0
            if my_agent.agent_view - my_agent.position[0] >= 0: # Top end
                signal[0:my_agent.agent_view - my_agent.position[0], :] = -1
            if my_agent.agent_view - my_agent.position[1] >= 0: # Left end
                signal[:, 0:my_agent.agent_view - my_agent.position[1]] = -1
            if self.position_state.region - my_agent.position[0] - my_agent.agent_view - 1 < 0:
                # Bottom end
                signal[
                    self.position_state.region - my_agent.position[0] - my_agent.agent_view - 1:,
                    :
                ] = -1
            if self.position_state.region - my_agent.position[1] - my_agent.agent_view - 1 < 0:
                # Right end
                signal[
                    :, self.position_state.region - my_agent.position[1] - my_agent.agent_view - 1:
                ] = -1

            # --- Determine the positions of all the other alive agents --- #
            for other_id, other_agent in self.agents.items():
                if other_id == my_agent.id: continue # Don't observe yourself
                if not other_agent.is_alive: continue # Can only observe alive agents
                r_diff = other_agent.position[0] - my_agent.position[0]
                c_diff = other_agent.position[1] - my_agent.position[1]
                if -my_agent.agent_view <= r_diff <= my_agent.agent_view and \
                        -my_agent.agent_view <= c_diff <= my_agent.agent_view:
                    r_diff += my_agent.agent_view
                    c_diff += my_agent.agent_view
                    signal[r_diff, c_diff] = 1 # There is an agent at this location.

            return {'position': signal}
        else:
            return {}


class GridPositionTeamBasedObserver:
    """
    Agents observe a grid of size agent_view centered on their
    position. The observation contains one channel per team, where the value of
    the cell is the number of agents on that team that occupy that square. -1
    indicates out of bounds.

    position (PositionState):
        The position state handler, which contains the region.

    number_of_teams (int):
        The number of teams in this simuation.
        Default 0.

    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position_state=None, number_of_teams=0, agents=None, **kwargs):
        self.position_state = position_state
        self.number_of_teams = number_of_teams + 1
        self.agents = agents

        for agent in self.agents.values():
            if isinstance(agent, AgentObservingAgent) and \
               isinstance(agent, PositionObservingAgent):
                agent.observation_space['position'] = Box(
                    -1,
                    len(self.agents),
                    (agent.agent_view*2+1, agent.agent_view*2+1, self.number_of_teams),
                    np.int
                )

    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position. Each team has its own channel and the value represents the number
        of agents of that team occupying the same square.
        """
        if isinstance(my_agent, AgentObservingAgent) and \
           isinstance(my_agent, PositionObservingAgent):
            signal = np.zeros((my_agent.agent_view*2+1, my_agent.agent_view*2+1))

            # --- Determine the boundaries of the agents' grids --- #
            # For left and top, we just do: view - x,y >= 0
            # For the right and bottom, we just do region - x,y - 1 - view > 0
            if my_agent.agent_view - my_agent.position[0] >= 0: # Top end
                signal[0:my_agent.agent_view - my_agent.position[0], :] = -1
            if my_agent.agent_view - my_agent.position[1] >= 0: # Left end
                signal[:, 0:my_agent.agent_view - my_agent.position[1]] = -1
            if self.position_state.region - my_agent.position[0] - my_agent.agent_view - 1 < 0:
                # Bottom end
                signal[
                    self.position_state.region - my_agent.position[0] - my_agent.agent_view - 1:,
                    :
                ] = -1
            if self.position_state.region - my_agent.position[1] - my_agent.agent_view - 1 < 0:
                # Right end
                signal[
                    :,
                    self.position_state.region - my_agent.position[1] - my_agent.agent_view - 1:
                ] = -1

            # Repeat the boundaries signal for all teams
            signal = np.repeat(signal[:, :, np.newaxis], self.number_of_teams, axis=2)

            # --- Determine the positions of all the other alive agents --- #
            for other_id, other_agent in self.agents.items():
                if other_id == my_agent.id: continue # Don't observe yourself
                if not other_agent.is_alive: continue # Can only observe alive agents
                r_diff = other_agent.position[0] - my_agent.position[0]
                c_diff = other_agent.position[1] - my_agent.position[1]
                if -my_agent.agent_view <= r_diff <= my_agent.agent_view and \
                        -my_agent.agent_view <= c_diff <= my_agent.agent_view:
                    r_diff += my_agent.agent_view
                    c_diff += my_agent.agent_view
                    signal[r_diff, c_diff, other_agent.team] += 1

            return {'position': signal}
        else:
            return {}


class SpeedObserver(Observer):
    """
    Observe the speed of all the agents in the simulator.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_obs_space(
            SpeedAngleObservingAgent,
            SpeedAngleAgent,
            lambda other: Box(-1, other.max_speed, (1,)),
            lambda: Box(-1, -1, (1,)), **kwargs
        )

    def get_obs(self, agent, **kwargs):
        """
        Get the speed of all the agents in the simulator.
        """
        return self._get_obs(
            agent,
            instance=SpeedAngleObservingAgent,
            other_instance=SpeedAngleAgent,
            attr='speed',
            **kwargs
        )

    @property
    def channel(self):
        return 'speed'

    @property
    def null_value(self):
        return -1


class AngleObserver(Observer):
    """
    Observe the angle of all the agents in the simulator.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_obs_space(
            SpeedAngleObservingAgent,
            SpeedAngleAgent,
            lambda *args: Box(-1, 360, (1,)),
            lambda *args: Box(-1, -1, (1,)), **kwargs
        )

    def get_obs(self, agent, **kwargs):
        """
        Get the angle of all the agents in the simulator.
        """
        return self._get_obs(
            agent,
            instance=SpeedAngleObservingAgent,
            other_instance=SpeedAngleAgent,
            attr='ground_angle',
            **kwargs
        )

    @property
    def channel(self):
        return 'ground_angle'

    @property
    def null_value(self):
        return -1


class VelocityObserver(Observer):
    """
    Observe the velocity of all the agents in the simulator.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_obs_space(
            VelocityObservingAgent, VelocityAgent,
            lambda other: Box(-other.max_speed, other.max_speed, (2,)),
            lambda: Box(0, 0, (2,)), **kwargs
        )

    def get_obs(self, agent, **kwargs):
        """
        Get the velocity of all the agents in the simulator.
        """
        return self._get_obs(
            agent,
            instance=VelocityObservingAgent,
            other_instance=VelocityAgent,
            attr='velocity',
            **kwargs
        )

    @property
    def channel(self):
        return 'velocity'

    @property
    def null_value(self):
        return np.zeros(2)


# -------------------------------- #
# --- Resources and Harvesting --- #
# -------------------------------- #

class GridResourceObserver:
    """
    Agents observe a grid of size resource_view centered on their
    position. The values in the grid are the values of the resources in that
    area.

    resources (ResourceState):
        The resource state handler.

    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, resource_state=None, agents=None, **kwargs):
        self.resource_state = resource_state
        self.agents = agents

        for agent in agents.values():
            if isinstance(agent, ResourceObservingAgent):
                agent.observation_space['resources'] = Box(
                    -1, self.resource_state.max_value,
                    (agent.resource_view*2+1, agent.resource_view*2+1)
                )

    def get_obs(self, agent, **kwargs):
        """
        These cells are filled with the values of the resources surrounding the
        agent's position.
        """
        if isinstance(agent, ResourceObservingAgent):
            signal = -np.ones((agent.resource_view*2+1, agent.resource_view*2+1))

            # Derived by considering each square in the resources as an "agent" and
            # then applied the agent diff logic from above. The resulting for-loop
            # can be written in the below vectorized form.
            (r, c) = agent.position
            r_lower = max([0, r-agent.resource_view])
            r_upper = min([self.resource_state.region-1, r+agent.resource_view])+1
            c_lower = max([0, c-agent.resource_view])
            c_upper = min([self.resource_state.region-1, c+agent.resource_view])+1
            signal[
                (r_lower+agent.resource_view-r):(r_upper+agent.resource_view-r),
                (c_lower+agent.resource_view-c):(c_upper+agent.resource_view-c)
            ] = self.resource_state.resources[r_lower:r_upper, c_lower:c_upper]
            return {'resources': signal}
        else:
            return {}


# ------------ #
# --- Team --- #
# ------------ #

class TeamObserver(Observer):
    """
    Observe the team of each agent in the simulator.
    """
    def __init__(self, number_of_teams=0, **kwargs):
        super().__init__(**kwargs)
        self.number_of_teams = number_of_teams
        self._set_obs_space_simple(
            TeamObservingAgent, lambda *args: Box(-1, self.number_of_teams, (1,), np.int), **kwargs
        )

    def get_obs(self, agent, **kwargs):
        """
        Get the team of each agent in the simulator.
        """
        return self._get_obs(
            agent,
            instance=TeamObservingAgent,
            attr='team',
            **kwargs
        )

    @property
    def channel(self):
        return 'team'

    @property
    def null_value(self):
        return -1
