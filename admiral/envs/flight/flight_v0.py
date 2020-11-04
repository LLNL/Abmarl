
import gym
from gym.spaces import Box
import numpy as np

class FlightEnv_v0(gym.Env):
    def __init__(self, config):
        """
        Initialize the birds flight environment. 
        """
        self.birds = config['birds']
        self.region = config['region']
        self.max_speed = config['max_speed']
        self.min_speed = config['min_speed']
        self.acceleration = config['acceleration']
        self.max_relative_angle = config['max_relative_angle']
        self.max_relative_angle_change = config['max_relative_angle_change']
        self.collision_distance = config['collision_distance']
        self.max_steps = config['max_steps']

        # Observe every bird's x-location, y-location, speed, ground-angle, and relative-angle.
        low_obs = np.repeat(
            np.array([
                -self.region, # minimum x position
                -self.region, # minimum y position
                self.min_speed, # minimum speed
                0.0, # minimum ground angle
                -self.max_relative_angle # minimum relative angle
            ]),
            self.birds # Repeat this "birds" time
        )
        high_obs = np.repeat(
            np.array([
                self.region, # maximum x position
                self.region, # maximum y position
                self.max_speed, # maximum speed
                360.0, # maximum ground angle
                self.max_relative_angle # maximum relative angle
            ]),
            self.birds # Repeat this "birds" time
        )
        self.observation_space = Box(low=low_obs, high=high_obs, dtype=np.float)
        
        # Each bird can accelerate/decelerate and change its angle.
        low_action = np.repeat(
            np.array([
                -self.acceleration, # minimum acceleration
                -self.max_relative_angle_change # minimum angle change
            ]),
            self.birds # Repeate this "birds" time
        )
        high_action = np.repeat(
            np.array([
                self.acceleration, # maximum acceleration
                self.max_relative_angle_change # maximum angle change
            ]),
            self.birds # Repeate this "birds" time
        )
        self.action_space = Box(low=low_action, high=high_action, dtype=np.float)

    def reset(self):
        """
        Uniformly distribute the birds throughout the central half of interest and assign them
        speeds, angles, and relative angles.

        Returns:
            (dict): The state of the environment.
        """
        self.step_count = 0
        self.x_position = np.random.uniform(-self.region/2, self.region/2, size=(self.birds,))
        self.y_position = np.random.uniform(-self.region/2, self.region/2, size=(self.birds,))
        self.speed = np.random.uniform(self.min_speed, self.max_speed, size=(self.birds,))
        self.ground_angle = np.random.uniform(0, 360, size=(self.birds,))
        self.relative_angle = np.random.uniform(-self.max_relative_angle, self.max_relative_angle, \
            size=(self.birds,))
        return self._collect_state()

    def step(self, action):
        """
        Move the birds according to the specifed action. The birds can accelerate/decelerate
        and change their angles. The environment takes these changes into account and propagates
        the birds forward. If a bird collides with another or a with a wall, the simulation ends.
        The agent is rewarded based on how long all the birds stay alive.

        Parameters:
            (np.array) action: An array of shape self.action_space that tells the birds how to
                accelerate/decelerate and adjust flight angles.
        
        Return:
            (state, reward, done_condition, info): A tuple with this relevant information.
        """
        self.step_count += 1

        # Propagate positions
        self.speed += action[:self.birds]
        self.speed[self.speed > self.max_speed] = self.max_speed
        self.speed[self.speed < self.min_speed] = self.min_speed

        self.relative_angle += action[self.birds:]
        self.relative_angle[self.relative_angle > self.max_relative_angle] = self.max_relative_angle
        self.relative_angle[self.relative_angle < -self.max_relative_angle] = -self.max_relative_angle
        self.ground_angle = (self.ground_angle + self.relative_angle) % 360

        self.x_position += self.speed*np.cos(np.deg2rad(self.ground_angle))
        self.y_position += self.speed*np.sin(np.deg2rad(self.ground_angle))

        # Calculate collision
        done = self._calculate_collision() or self.step_count >= self.max_steps

        return self._collect_state(), 1, done, {}

    def render(self, *args, **kwargs):
        """
        Render the environment.
        """
        import matplotlib.pyplot as plt
        plt.gcf().clear()
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.scatter(self.x_position, self.y_position, s=75, c = 'w', marker='o', edgecolor='b', \
            alpha=0.9)
        plt.xlim([-self.region, self.region])
        plt.ylim([-self.region, self.region])
        plt.draw()
        plt.pause(1e-17)

    def _calculate_collision(self):
        """
        Calculate if a collision happens with the edge or with another bird. This is its own
        function because we need to break out of a nested loop, which is most easily done with a
        return.

        Return:
            (bool): Returns the done condition, which indicates if a collision has occured.
        """
        # Calculate collision with wall
        if np.any(self.x_position < -self.region + self.collision_distance) \
                or np.any(self.x_position > self.region - self.collision_distance) \
                or np.any(self.y_position < -self.region + self.collision_distance) \
                or np.any(self.y_position > self.region - self.collision_distance):
            return True
        
        # Calculate collision with other birds
        for i in range(self.birds):
            for j in range(i+1,self.birds):
                dist = np.sqrt(np.square(self.x_position[i] - self.x_position[j]) \
                    + np.square(self.y_position[i] - self.y_position[j]))
                if dist < self.collision_distance:
                    return True
        
        return False
    
    def _collect_state(self):
        """
        Consolidate the different class attributes that make up the state.
        """
        return np.concatenate((
            self.x_position,
            self.y_position,
            self.speed,
            self.ground_angle,
            self.relative_angle
        ))
