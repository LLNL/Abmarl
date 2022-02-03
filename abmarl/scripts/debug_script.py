def create_parser(subparsers):
    """Parse the arguments for the debug command.

    Returns:
        parser: ArgumentParser for debug command.
    """
    debug_parser = subparsers.add_parser(
        'debug',
        help="Run the simulation from the configruation file with random "
             "actions, rendering each step and outputting observations and actions. "
             "This is useful to ensure that you've setup the simulation correctly."
    )
    debug_parser.add_argument(
        'configuration', type=str, help='Path to python config file. Include the .py extension.'
    )
    debug_parser.add_argument(
        '-n', '--episodes', type=int, default=1, help='The number of episodes to run. Default 1.'
    )
    debug_parser.add_argument(
        '-s', '--steps-per-episode', type=int, default=200,
        help='The maximum number of steps to take per epsiode. Default 200.'
    )
    debug_parser.add_argument(
        '--render', action='store_true', help='Render the simulation each step.'
    )
    return debug_parser


def run(full_config_path, parameters):
    from abmarl import debug
    debug.run(full_config_path, parameters)
