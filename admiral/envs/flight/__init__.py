from .flight_v0 import FlightEnv_v0

from numbers import Number
import warnings

def build_flight_v0(env_config={}):
    """
    Flight model builder. Creates an instantiation of Flight object with the environment
    configuration.

    Parameters:
        (int) birds: The number of birds. Default 2.
        (float) region: The size of the region of interest. Default 25.
        (float) max_speed: The maximum speed that each bird can travel, where speed is
            measured as the Euclidean norm of the velocity. Default 1.0.
        (float) min_speed: The minimum speed that each bird can travel, where speed is
            measured as the Euclidean norm of the velocity. Default 0.5.
        (float) acceleration: The maximum acceleration of the bird. Default 0.1.
        (float) angle: The maximum flight angle the bird can reach. The location of the
            bird at the next time step depends on its velocity and angle. Default 10.0 degrees.
        (float) angle_change: The maximum change in flight angle. Default 5.0.
        (float) collision_distance: The distance between two birds or between a bird and
            the edge of the map that is considered a collision. Default 1.0
        (int) max_steps: The maximum number of steps per epsidoe. Default 200
    
    Return:
        (object): an instantiation of the Flight environment.
    """
    config = {
        'birds': 2,
        'region': 25.0,
        'max_speed': 1.0,
        'min_speed': 0.5,
        'acceleration': 0.1,
        'max_relative_angle': 10.0,
        'max_relative_angle_change': 5.0,
        'collision_distance': 1.0,
        'max_steps': 200,
    }

    if 'birds' in env_config:
        if type(env_config['birds']) is not int:
            warnings.warn('Birds must be an integer. Using default value: ' + str(config['birds']))
        elif env_config['birds'] < 1:
            warnings.warn('You must have at least 1 bird. Using default value: ' + str(config['birds']))
        else:
            config['birds'] = env_config['birds']

    if 'region' in env_config:
        if not isinstance(env_config['region'], Number):
            warnings.warn('Region must be a number. Using default value: ' + str(config['region']))
        elif env_config['region'] < 0:
            warnings.warn('Region must be positive. Using default value: ' + str(config['region']))
        else:
            config['region'] = env_config['region']

    if 'max_speed' in env_config:
        if not isinstance(env_config['max_speed'], Number):
            warnings.warn('Max speed must be a number. Using default value: ' + str(config['max_speed']))
        elif env_config['max_speed'] <= 0:
            warnings.warn('Max speed must be positive. Using default value: ' + str(config['max_speed']))
        else:
            config['max_speed'] = env_config['max_speed']

    if 'min_speed' in env_config:
        if not isinstance(env_config['min_speed'], Number):
            warnings.warn('Min speed must be a number. Using default value: ' + str(config['min_speed']))
        elif env_config['min_speed'] < 0:
            warnings.warn('Min speed must be nonnegative. Using default value: ' + str(config['min_speed']))
        else:
            config['min_speed'] = env_config['min_speed']

    if 'acceleration' in env_config:
        if not isinstance(env_config['acceleration'], Number):
            warnings.warn('Acceleration must be a number. Using default value: ' + str(config['acceleration']))
        elif env_config['acceleration'] < 0:
            warnings.warn('Acceleration must be nonnegative. Using default value: ' + str(config['acceleration']))
        else:
            config['acceleration'] = env_config['acceleration']

    if 'max_relative_angle' in env_config:
        if not isinstance(env_config['max_relative_angle'], Number):
            warnings.warn('Max relative angle must be a number. Using default value: ' + str(config['max_relative_angle']))
        elif env_config['max_relative_angle'] < 0:
            warnings.warn('Max relative angle must be nonnegative. Using default value: ' + str(config['max_relative_angle']))
        else:
            config['max_relative_angle'] = env_config['max_relative_angle']

    if 'max_relative_angle_change' in env_config:
        if not isinstance(env_config['max_relative_angle_change'], Number):
            warnings.warn('Max relative angle change must be a number. Using default value: ' + str(config['max_relative_angle_change']))
        elif env_config['max_relative_angle_change'] < 0:
            warnings.warn('Max relative angle change must be nonnegative. Using default value: ' + str(config['max_relative_angle_change']))
        else:
            config['max_relative_angle_change'] = env_config['max_relative_angle_change']

    if 'collision_distance' in env_config:
        if not isinstance(env_config['collision_distance'], Number):
            warnings.warn('Collision distance must be a number. Using default value: ' + str(config['collision_distance']))
        elif env_config['collision_distance'] < 0:
            warnings.warn('Collision distance must be nonnegative. Using default value: ' + str(config['collision_distance']))
        else:
            config['collision_distance'] = env_config['collision_distance']

    if 'max_steps' in env_config:
        if type(env_config['max_steps']) is not int:
            warnings.warn('Max steps must be an integer. Using default value: ' + str(config['max_steps']))
        elif env_config['max_steps'] < 1:
            warnings.warn('Max steps must be at least 1 bird. Using default value: ' + str(config['max_steps']))
        else:
            config['max_steps'] = env_config['max_steps']

    return FlightEnv_v0(config)

