
import numpy as np

from admiral.envs.components.agent import AttackingAgent, GridMovementAgent, HarvestingAgent, \
    SpeedAngleAgent, AcceleratingAgent, LifeAgent, TeamAgent, PositionAgent, VelocityAgent

# ----------------- #
# --- Attacking --- #
# ----------------- #

class PositionBasedAttackActor:
    """
    Provide the necessary action space for agents who can attack and processes such
    attacks. The attack is successful if the attacked agent is alive and within
    range. The action space is appended with a MultiBinary(1), allowing the agent
    to attack or not attack.

    agents (dict):
        The dictionary of agents. Because attacks are distance-based, all agents must
        be PositionAgents; becuase the attacked agent must be alive, all agents
        must be LifeAgents.
    
    attack_norm (int):
        Norm used to measure the distance between agents. For example, you might
        use a norm of 1 or np.inf in a Gird space, while 2 might be used in a Contnuous
        space. Default is np.inf.
    """
    def __init__(self, agents=None, attack_norm=np.inf, **kwargs):
        assert type(agents) is dict, "agents must be a dict"
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
            assert isinstance(agent, LifeAgent)
        self.agents = agents

        from gym.spaces import MultiBinary
        for agent in self.agents.values():
            if isinstance(agent, AttackingAgent):
                agent.action_space['attack'] = MultiBinary(1)
        
        self.attack_norm = attack_norm

    def process_attack(self, attacking_agent, attack, **kwargs):
        """
        Determine which agent the attacking agent successfully attacks.

        attacking_agent (AttackingAgent):
            The agent that we are processing.

        attack (bool):
            True if the agent has chosen to attack, otherwise False.

        return (Agent):
            Return the attacked agent object. This can be None if no agent was
            attacked.
        """
        if isinstance(attacking_agent, AttackingAgent) and attack:
            for attacked_agent in self.agents.values():
                if attacked_agent.id == attacking_agent.id:
                    # Cannot attack yourself
                    continue
                elif not attacked_agent.is_alive:
                    # Cannot attack dead agents
                    continue
                elif np.linalg.norm(attacking_agent.position - attacked_agent.position, self.attack_norm) > attacking_agent.attack_range:
                    # Agent too far away
                    continue
                else:
                    return attacked_agent

class PositionTeamBasedAttackActor:
    """
    Provide the necessary action space for agents who can attack and process such
    attacks. The attack is successful if the attacked agent is alive, on a different
    team, and within range. The action space is appended with a MultiBinary(1),
    allowing the agent to attack or not attack.

    agents (dict):grid
        The dictionary of agents. Because attacks are distance-based, all agents must
        be PositionAgents; because the attacked agent must be alive, all agents
        must be LifeAgents; and because the attacked agent must be on a different
        team, all agents must be TeamAgents.
    
    attack_norm (int):
        Norm used to measure the distance between agents. For example, you might
        use a norm of 1 or np.inf in a Gird space, while 2 might be used in a Contnuous
        space. Default is np.inf.
    """
    def __init__(self, agents=None, attack_norm=np.inf, **kwargs):
        assert type(agents) is dict, "agents must be a dict"
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
            assert isinstance(agent, TeamAgent)
            assert isinstance(agent, LifeAgent)
        self.agents = agents

        from gym.spaces import MultiBinary
        for agent in self.agents.values():
            if isinstance(agent, AttackingAgent):
                agent.action_space['attack'] = MultiBinary(1)
        
        self.attack_norm = attack_norm

    def process_attack(self, attacking_agent, attack, **kwargs):
        """
        Determine which agent the attacking agent successfully attacks.

        attacking_agent (AttackingAgent):
            The agent that we are processing.

        attack (bool):
            True if the agent has chosen to attack, otherwise False.

        return (Agent):
            Return the attacked agent object.
        """
        if isinstance(attacking_agent, AttackingAgent) and attack:
            for attacked_agent in self.agents.values():
                if attacked_agent.id == attacking_agent.id:
                    # Cannot attack yourself
                    continue
                elif not attacked_agent.is_alive:
                    # Cannot attack dead agents
                    continue
                elif attacking_agent.team == attacked_agent.team:
                    # Cannot attack agents on the same team
                    continue
                elif np.linalg.norm(attacking_agent.position - attacked_agent.position, self.attack_norm) > attacking_agent.attack_range:
                    # Agent too far away
                    continue
                else:
                    return attacked_agent



# ----------------------------- #
# --- Position and Movement --- #
# ----------------------------- #

class GridMovementActor:
    """
    Provides the necessary action space for agents who can move and processes such
    movements.

    position (GridPositionState):
        The position state handler. Needed to modify the agents' positions.

    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents = agents
        # Not all agents need to be PositionAgents; only the ones who can move.

        from gym.spaces import Box
        for agent in self.agents.values():
            if isinstance(agent, GridMovementAgent):
                agent.action_space['move'] = Box(-agent.move_range, agent.move_range, (2,), np.int)

    def process_move(self, moving_agent, move, **kwargs):
        """
        Determine the agent's new position based on its move action.

        moving_agent (GridMovementAgent):
            The agent that moves.
        
        move (np.array):
            How much the agent would like to move in row and column.
        
        return (np.array):
            How much the agent has moved in row and column. This can be different
            from the desired move if the position update was invalid.
        """
        if isinstance(moving_agent, GridMovementAgent) and isinstance(moving_agent, PositionAgent):
            position_before = moving_agent.position
            self.position.modify_position(moving_agent, move, **kwargs)
            return moving_agent.position - position_before

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

        from gym.spaces import Box
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

class AccelerationMovementActor:
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
    def __init__(self, position_state=None, velocity_state=None, agents=None, **kwargs):
        self.position_state = position_state
        self.velocity_state = velocity_state
        self.agents = agents

        from gym.spaces import Box
        for agent in agents.values():
            if isinstance(agent, AcceleratingAgent):
                agent.action_space['accelerate'] = Box(-agent.max_acceleration, agent.max_acceleration, (2,))
    
    def process_move(self, agent, acceleration, **kwargs):
        """
        Update the agent's velocity by applying the acceleration. Then use the
        updated velocity to determine the agent's next position.

        agent (AcceleratingAgent):
            Agent that is attempting to move.
        
        acceleration (np.array):
            A two-element float array that changes the agent's velocity. New velocity
            must be within the agent's max speed.
        
        return (np.array):
            Return the change in position.
        """
        # TODO: maybe these should check the state types, like velocity and position, instead of the actor type...
        if isinstance(agent, VelocityAgent) and isinstance(agent, PositionAgent):
            self.velocity_state.modify_velocity(agent, acceleration)
            position_before = agent.position
            self.position_state.modify_position(agent, agent.velocity, **kwargs)
            return agent.position - position_before
                


# -------------------------------- #
# --- Resources and Harvesting --- #
# -------------------------------- #

class GridResourcesActor:
    """
    Provides the necessary action space for agents who can harvest resources and
    processes the harvesting action.

    resources (ResourceState):
        The resource state handler.

    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, resources=None, agents=None, **kwargs):
        self.resources = resources
        self.agents = agents

        from gym.spaces import Box
        for agent in agents.values():
            if isinstance(agent, HarvestingAgent):
                agent.action_space['harvest'] = Box(0, agent.max_harvest, (1,), np.float)

    def process_harvest(self, agent, amount, **kwargs):
        """
        Harvest some amount of resources at the agent's position.

        agent (HarvestingAgent):
            The agent who has chosen to harvest the resource.

        amount (float):
            The amount of resource the agent wants to harvest.
        
        return (float):
            Return the amount of resources that was actually harvested. This can
            be less than the desired amount if the cell does not have enough resources.
        """
        if isinstance(agent, HarvestingAgent) and isinstance(agent, PositionAgent):
            location = tuple(agent.position)
            resource_before = self.resources.resources[location]
            self.resources.modify_resources(location, -amount)
            return resource_before - self.resources.resources[location]
