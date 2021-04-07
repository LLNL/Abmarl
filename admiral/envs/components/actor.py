from abc import ABC, abstractmethod, abstractproperty

from gym.spaces import Discrete, Box
import numpy as np

from admiral.envs.components.agent import AttackingAgent, GridMovementAgent, HarvestingAgent, \
    SpeedAngleAgent, AcceleratingAgent, LifeAgent, TeamAgent, PositionAgent, VelocityAgent, \
    CollisionAgent, BroadcastingAgent

class Actor(ABC):
    """
    Base actor class provides the interface required of all actors.
    """
    def __init__(self, agents=None, instance=None, space_func=None, **kwargs):
        self.agents = agents
        for agent in self.agents.values():
            if isinstance(agent, instance):
                agent.action_space[self.channel] = space_func(agent)
    
    @abstractmethod
    def process_action(self, agent, action_dict, **kwargs): pass

    def _get_action_from_dict(self, action_dict, **kwargs):
        return action_dict.get(self.channel, self.null_value)

    @abstractproperty
    def channel(self): pass

    @abstractproperty
    def null_value(self): pass

# ----------------- #
# --- Attacking --- #
# ----------------- #

class AttackActor(Actor):
    """
    Agents can attack other agents within their attack radius. If there are multiple
    attackable agents in the radius, then one will be randomly chosen. Attackable
    agents are determiend by the team_matrix.

    agents (dict of Agents):
        The dictionary of agents.
    
    attack_norm (int):
        Norm used to measure the distance between agents. For example, you might
        use a norm of 1 or np.inf in a Gird space, while 2 might be used in a Continuous
        space. Default is np.inf.
    
    team_attack_matrix (np.ndarray):
        A matrix that indicates which teams can attack which other team using the
        value at the index, like so:
            team_matrix[attacking_team, attacked_team] = 0 or 1.
        0 indicates it cannot attack, 1 indicates that it can.
        Default None, meaning that any team can attack any other team, and no team
        can attack itself.
    
    number_of_teams (int):
        Specify the number of teams in the simulation for building the team_attack_matrix
        if that is not specified here.
        Default 0, indicating that there are no teams and its a free-for-all battle.
    """
    def __init__(self, attack_norm=np.inf, team_attack_matrix=None, number_of_teams=0, **kwargs):
        super().__init__(
            isinstance=AttackingAgent,
            space_func=lambda agent: Discrete(2),
            **kwargs
        )
        if team_attack_matrix is None:
            # Default: teams can attack all other teams but not themselves. Agents
            # that are "on team 0" are actually teamless, so they can be attacked
            # by and can attack agents from any other team, including "team 0"
            # agents.
            self.team_attack_matrix = -np.diag(np.ones(number_of_teams+1)) + 1
            self.team_attack_matrix[0,0] = 1
        else:
            self.team_attack_matrix = team_attack_matrix
        self.attack_norm = attack_norm
    
    def process_action(self, attacking_agent, action_dict, **kwargs):
        """
        If the agent has chosen to attack, then determine which agent got attacked.

        return (Agent):
            Return the attacked agent object. This can be None if no agent was
            attacked.
        """
        if self._get_action_from_dict(action_dict):
            for attacked_agent in self.agents.values():
                if attacked_agent.id == attacking_agent.id:
                    # Cannot attack yourself
                    continue
                elif not attacked_agent.is_alive:
                    # Cannot attack a dead agent
                    continue
                elif np.linalg.norm(attacking_agent.position - attacked_agent.position, self.attack_norm) > attacking_agent.attack_range:
                    # Agent is too far away
                    continue
                elif not self.team_attack_matrix[attacking_agent.team, attacked_agent.team]:
                    # Attacking agent cannot attack this agent
                    continue
                elif np.random.uniform() > attacking_agent.attack_accuracy:
                    # Attempted attack, but it failed
                    continue
                else:
                    # The agent was successfully attacked!
                    return attacked_agent
    
    @property
    def channel(self):
        return 'attack'
    
    @property
    def null_value(self):
        return False



# --------------------- #
# --- Communication --- #
# --------------------- #

class BroadcastActor(Actor):
    """
    BroadcastingAgents can choose to broadcast in this step or not.

    broadcast_state (BroadcastState):
        The broadcast state handler. Needed to modifying the agents' broadcasting state.

    agents (dict):
        Dictionary of agents.
    """
    def __init__(self, broadcast_state=None, **kwargs):
        super().__init__(
            instance=BroadcastingAgent,
            space_func=lambda agent: Discrete(2),
            **kwargs
        )
        self.broadcast_state = broadcast_state
    
    def process_action(self, agent, action_dict, **kwargs):
        """
        Determine the agents new broadcasting state based on its action.

        return: bool
            The agent's broadcasting state.
        """
        broadcasting = self._get_action_from_dict(action_dict)
        self.broadcast_state.modify_broadcast(agent, broadcasting)
        return agent.broadcasting
        
    @property
    def channel(self):
        return 'broadcast'
    
    @property
    def null_value(self):
        return False



# ----------------------------- #
# --- Position and Movement --- #
# ----------------------------- #

class GridMovementActor(Actor):
    """
    Provides the necessary action space for agents who can move and processes such
    movements.

    position (GridPositionState):
        The position state handler. Needed to modify the agents' positions.

    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position_state=None, **kwargs):
        super().__init__(
            instance=GridMovementAgent,
            space_func=lambda agent: Box(-agent.move_range, agent.move_range, (2,), np.int),
            **kwargs
        )
        self.position_state = position_state

    def process_action(self, agent, action_dict, **kwargs):
        """
        Determine the agent's new position based on its move action.
        
        return (np.array):
            How much the agent has moved in row and column. This can be different
            from the desired move if the position update was invalid.
        """
        move = self._get_action_from_dict(action_dict)
        position_before = agent.position
        self.position.modify_position(agent, move, **kwargs)
        return agent.position - position_before
    
    @property
    def channel(self):
        return 'move'
    
    @property
    def null_value(self):
        return np.zeros(2)

# TODO: SpeedAngleMovementActor is the only one that doesn't follow the actor
# interface because it has two channels. Modify it so that it fits the interface.
class SpeedAngleMovementActor:
    """
    Process acceleration and angle changes for SpeedAngleAgents. Update the agents'
    positions based on their new speed and direction.

    position (ContinuousPositionState):
        The position state handler. Needed to modify agent positions.
    
    speed_angle (SpeedAngleState):
        The speed and angle state handler.

    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position=None, speed_angle=None, agents=None, **kwargs):
        self.position = position
        self.speed_angle = speed_angle
        self.agents = agents

        for agent in agents.values():
            if isinstance(agent, SpeedAngleAgent):
                agent.action_space['accelerate'] = Box(-agent.max_acceleration, agent.max_acceleration, (1,))
                agent.action_space['bank'] = Box(-agent.max_banking_angle_change, agent.max_banking_angle_change, (1,))
    
    def process_move(self, agent, acceleration, angle, **kwargs):
        """
        Update the agent's speed by applying the acceleration and the agent's banking
        angle by applying the change. Then use the updated speed and ground angle
        to determine the agent's next position.

        agent (SpeedAngleAgent):
            Agent that is attempting to move.
        
        acceleration (np.array):
            A one-element float array that changes the agent's speed. New speed
            must be within the agent's min and max speed.
        
        angle (np.array):
            A one-element float array that changes the agent's banking angle. New
            banking angle must be within the agent's min and max banking angles.

        return (np.array):
            Return the change in position.
        """
        if isinstance(agent, SpeedAngleAgent):
            self.speed_angle.modify_speed(agent, acceleration[0])
            self.speed_angle.modify_banking_angle(agent, angle[0])
            
            x_position = agent.speed*np.cos(np.deg2rad(agent.ground_angle))
            y_position = agent.speed*np.sin(np.deg2rad(agent.ground_angle))
            
            position_before = agent.position
            self.position.modify_position(agent, np.array([x_position, y_position]))
            return agent.position - position_before

class AccelerationMovementActor(Actor):
    """
    Process x,y accelerations for AcceleratingAgents, which are given an 'accelerate'
    action. Update the agents' positions based on their new velocity.

    position_state (ContinuousPositionState):
        The position state handler. Needed to modify agent positions.
    
    velocity_state (VelocityState):
        The velocity state handler. Needed to modify agent velocities.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position_state=None, velocity_state=None, **kwargs):
        super().__init__(
            instance=AcceleratingAgent,
            space_func=lambda agent: Box(-agent.max_acceleration, agent.max_acceleration, (2,)),
            **kwargs
        )
        self.position_state = position_state
        self.velocity_state = velocity_state
    
    def process_action(self, agent, action_dict, **kwargs):
        """
        Update the agent's velocity by applying the acceleration. Then use the
        updated velocity to determine the agent's next position.
        
        return (np.array):
            Return the change in position.
        """
        acceleration = self._get_action_from_dict(action_dict)
        self.velocity_state.modify_velocity(agent, acceleration)
        position_before = agent.position
        self.position_state.modify_position(agent, agent.velocity, **kwargs)
        return agent.position - position_before
        
    @property
    def channel(self):
        return 'accelerate'
    
    @property
    def null_value(self):
        return np.zeros(2)
                


# -------------------------------- #
# --- Resources and Harvesting --- #
# -------------------------------- #

class GridResourcesActor(Actor):
    """
    Provides the necessary action space for agents who can harvest resources and
    processes the harvesting action.

    resources (ResourceState):
        The resource state handler.

    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, resource_state=None, **kwargs):
        super().__init__(
            instance=HarvestingAgent,
            space_func=lambda agent: Box(0, agent.max_harvest, (1,)),
            **kwargs
        )
        self.resource_state = resource_state

    def process_action(self, agent, action_dict, **kwargs):
        """
        Harvest some amount of resources at the agent's position.
        
        return (float):
            Return the amount of resources that was actually harvested. This can
            be less than the desired amount if the cell does not have enough resources.
        """
        amount = self._get_action_from_dict(action_dict)
        location = tuple(agent.position)
        resource_before = self.resource_state.resources[location]
        self.resource_state.modify_resources(location, -amount)
        return resource_before - self.resource_state.resources[location]
    
    @property
    def channel(self):
        return 'harvest'
    
    @property
    def null_value(self):
        return 0





# --------------------------------------------- #
# --- Actors that don't receive agent input --- #
# --------------------------------------------- #

class ContinuousCollisionActor:
    """
    Identify collisions among agents and update positions and velocities according to
    elastic collision physics.

    position_state (PositionState):
        The PositionState handler.
    
    velocity_state (VelocityState):
        The VelocityState handler.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position_state=None, velocity_state=None, agents=None, **kwargs):
        self.position_state = position_state
        self.velocity_state = velocity_state
        self.agents = agents

    def detect_collisions_and_modify_states(self, **kwargs):
        """
        Detect collisions between agents and update position and velocities.
        """
        checked_agents = set()
        for agent1 in self.agents.values():
            if not (isinstance(agent1, CollisionAgent) and isinstance(agent1, PositionAgent) and isinstance(agent1, VelocityAgent)): continue
            checked_agents.add(agent1.id)
            for agent2 in self.agents.values():
                if not (isinstance(agent2, PositionAgent) and isinstance(agent1, VelocityAgent) and isinstance(agent2, CollisionAgent)): continue
                if agent1.id == agent2.id: continue # Cannot collide with yourself
                if agent2.id in checked_agents: continue # Already checked this agent
                dist = np.linalg.norm(agent1.position - agent2.position)
                combined_sizes = agent1.size + agent2.size
                if dist < combined_sizes:
                    self._undo_overlap(agent1, agent2, dist, combined_sizes)
                    self._update_velocities(agent1, agent2)

    def _undo_overlap(self, agent1, agent2, dist, combined_sizes, **kwargs):
        """
        Colliding agents can overlap within a timestep. So we need to move the
        colliding agents "backwards" through their path in order to capture the
        positions they were in when they actually collided.

        agent1 (CollisionAgent):
            One of the colliding agents.
        
        agent2 (CollisionAgent):
            The other colliding agent.
        
        dist (float):
            The collision distance threshold.
        
        combined_size (float):
            The combined size of the two agents
        """
        overlap = (combined_sizes - dist) / combined_sizes
        self.position_state.modify_position(agent1, -agent1.velocity * overlap)
        self.position_state.modify_position(agent2, -agent2.velocity * overlap)

    def _update_velocities(self, agent1, agent2, **kwargs):
        """
        Updates the velocities of two agents when they collide based on an
        elastic collision assumption.

        agent1 (CollisionAgent):
            One of the colliding agents.
        
        agent2 (CollisionAgent):
            The other colliding agent.
        """
        # calculate vector between centers
        rel_position = [
            agent2.position - agent1.position,
            agent1.position - agent2.position
        ]
        # Calculate relative velocities
        rel_velocities = [
            agent1.velocity - agent2.velocity,
            agent2.velocity - agent1.velocity
        ]
        # Calculate mass factor
        mass_factor = [
            2 * agent2.mass / (agent2.mass + agent1.mass),
            2 * agent1.mass / (agent2.mass + agent1.mass)
        ]
        # norm
        norm = [
            np.square(np.linalg.norm(rel_position[0])),
            np.square(np.linalg.norm(rel_position[1]))
        ]
        # Dot product of relative velocity and relative distcance
        dot = [
            np.dot(rel_velocities[0], rel_position[0]),
            np.dot(rel_velocities[1], rel_position[1])
        ]
        # bringing it all together
        vel_new = [
            agent1.velocity - (mass_factor[0] * (dot[0]/norm[0]) * rel_position[0]),
            agent2.velocity - (mass_factor[1] * (dot[1]/norm[1]) * rel_position[1])
        ]
        # Only update the velocity if not stationary
        self.velocity_state.set_velocity(agent1, vel_new[0])
        self.velocity_state.set_velocity(agent2, vel_new[1])
