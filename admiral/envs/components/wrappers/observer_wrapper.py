# Demoing an observation wrapper design for partial observations #

import numpy as np

from admiral.envs.components.observer import HealthObserver, LifeObserver, MaskObserver, \
    PositionObserver, RelativePositionObserver, SpeedObserver, AngleObserver, VelocityObserver, \
    TeamObserver
# TODO: implement for grid-based observers too, like the resource observer

from admiral.envs.components.agent import PositionAgent

class PositionRestrictedObservationWrapper:
    def __init__(self, observer, obs_filter=None, obs_norm=np.inf, agents=None, **kwargs):
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
                    self.obs_filter(np.linalg.norm(agent.position - other.position, self.obs_norm)): \
                        continue # We perfectly observed this agent
                else:
                    # We did not observe the agent, so we have to modify the obs
                    for obs_type, obs_content in obs.items():
                        obs_content[other.id] = self.observer.null_value
        return obs



# Test it
from admiral.envs.components.observer import HealthObserver, LifeObserver, MaskObserver, \
    PositionObserver, RelativePositionObserver, SpeedObserver, AngleObserver, VelocityObserver, \
    TeamObserver
from admiral.envs.components.state import LifeState, GridPositionState, SpeedAngleState, VelocityState, TeamState
from admiral.envs.components.agent import PositionAgent, LifeAgent, TeamAgent, SpeedAngleAgent, VelocityAgent, \
    PositionObservingAgent, LifeObservingAgent, HealthObservingAgent, TeamObservingAgent, SpeedAngleObservingAgent, VelocityObservingAgent

class TestAgent(
    PositionAgent, PositionObservingAgent, \
    LifeAgent, LifeObservingAgent, HealthObservingAgent, \
    TeamAgent, TeamObservingAgent, \
    SpeedAngleAgent, SpeedAngleObservingAgent, \
    VelocityAgent, VelocityObservingAgent
): pass

agents = {
    'agent0': TestAgent(id='agent0', initial_position=np.array([2, 2]), initial_health=0.67, team=0, initial_speed=0.30, initial_banking_angle=7,  initial_ground_angle=123, initial_velocity=np.array([-0.3, 0.8])),
    'agent1': TestAgent(id='agent1', initial_position=np.array([4, 4]), initial_health=0.54, team=1, initial_speed=0.00, initial_banking_angle=0,  initial_ground_angle=126, initial_velocity=np.array([-0.2, 0.7])),
    'agent2': TestAgent(id='agent2', initial_position=np.array([4, 3]), initial_health=0.36, team=1, initial_speed=0.12, initial_banking_angle=30, initial_ground_angle=180, initial_velocity=np.array([-0.1, 0.6])),
    'agent3': TestAgent(id='agent3', initial_position=np.array([4, 2]), initial_health=0.24, team=1, initial_speed=0.05, initial_banking_angle=13, initial_ground_angle=46,  initial_velocity=np.array([0.0,  0.5])),
    'agent4': TestAgent(id='agent4', initial_position=np.array([4, 1]), initial_health=0.12, team=2, initial_speed=0.17, initial_banking_angle=15, initial_ground_angle=212, initial_velocity=np.array([0.1,  0.4])),
    'agent5': TestAgent(id='agent5', initial_position=np.array([4, 0]), initial_health=0.89, team=1, initial_speed=0.21, initial_banking_angle=23, initial_ground_angle=276, initial_velocity=np.array([0.2,  0.3])),
    'agent6': TestAgent(id='agent6', initial_position=np.array([1, 1]), initial_health=0.53, team=0, initial_speed=0.39, initial_banking_angle=21, initial_ground_angle=300, initial_velocity=np.array([0.3,  0.2])),
    'agent7': TestAgent(id='agent7', initial_position=np.array([0, 4]), initial_health=0.50, team=1, initial_speed=0.36, initial_banking_angle=24, initial_ground_angle=0,   initial_velocity=np.array([0.4,  0.1])),
    'agent8': TestAgent(id='agent8', initial_position=np.array([2, 1]), initial_health=0.26, team=0, initial_speed=0.06, initial_banking_angle=16, initial_ground_angle=5,   initial_velocity=np.array([0.5,  0.0])),
    'agent9': TestAgent(id='agent9', initial_position=np.array([2, 2]), initial_health=0.08, team=2, initial_speed=0.24, initial_banking_angle=30, initial_ground_angle=246, initial_velocity=np.array([0.6, -0.1])),
}

def drop_off_line(distance):
    slope = 3
    return 1. - 1. / (slope) * distance

def no_obs(distance):
    return 0

position_state = GridPositionState(region=5, agents=agents)
life_state = LifeState(agents=agents)
speed_angle_state = SpeedAngleState(agents=agents)
velocity_state = VelocityState(agents=agents)
team_state = TeamState(agents=agents, number_of_teams=3)

position_state.reset()
life_state.reset()
speed_angle_state.reset()
velocity_state.reset()

health_observer = HealthObserver(agents=agents)
life_observer = LifeObserver(agents=agents)
mask_observer = MaskObserver(agents=agents)
position_observer = PositionObserver(position=position_state, agents=agents)
relative_position_observer = RelativePositionObserver(position=position_state, agents=agents)
speed_observer = SpeedObserver(agents=agents)
angle_observer = AngleObserver(agents=agents)
velocity_observer = VelocityObserver(agents=agents)
team_observer = TeamObserver(team=team_state, agents=agents)

wrapped_health_observer = PositionRestrictedObservationWrapper(health_observer, obs_filter=drop_off_line, agents=agents)
print(wrapped_health_observer.get_obs(agents['agent0']))

