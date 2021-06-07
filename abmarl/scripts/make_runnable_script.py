def create_parser(subparsers):
    """Parse the arguments for the make_runnable command.

    Returns
    -------
        parser : ArgumentParser
    """
    runnable_parser = subparsers.add_parser(
        'make-runnable',
        help='Convert a configuration + script to be runnable from the commandline. Save the '
        'coverted script in the same directory as the original script. This option is useful for '
        'running batch jobs from the command line and/or integration with magpie.'
    )
    runnable_parser.add_argument(
        'configuration', type=str, help='Path to python config file. Include the .py extension.'
    )
    runnable_parser.add_argument(
        '--magpie', action='store_true', help='Output a magpie script as well.'
    )
    runnable_parser.add_argument(
        '-n', '--nodes', type=int,
        help='Specify the number of compute nodes (not rollout workers) you want.', default=2
    )
    runnable_parser.add_argument(
        '-t', '--time-limit', type=str, help='The maximum runtime for this job in hours',
        default='2'
    )
    return runnable_parser


def run(full_config_path, parameters):
    from abmarl import make_runnable
    make_runnable.run(full_config_path, parameters)
