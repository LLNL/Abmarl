def create_parser(subparsers):
    """Parse the arguments for the scale command.

    Returns
    -------
        parser : ArgumentParser
    """
    scale_parser = subparsers.add_parser(
        'scale',
        help="Create scripts to enable a training configruation to be trained at "
        "scale using slurm with RLlib's client-server model and configuration."
    )
    scale_parser.add_argument(
        'configuration', type=str, help='Path to python config file. Include the .py extension.'
    )
    scale_parser.add_argument(
        '-n', '--nodes', type=int,
        help='The number of compute nodes to use in this experiment. The first '
        'node is the server and all the rest are clients.',
        default=2
    )
    scale_parser.add_argument(
        '-t', '--time', type=int, help='The maximum runtime for this job in hours',
        default=2
    )
    scale_parser.add_argument(
        '-p', '--partition', type=str, help='The partition on which to run this job.',
        default='pbatch'
    )
    scale_parser.add_argument(
        '--virtual-env-path', type=str, help='Full path to the python virtual environment.'
    )
    scale_parser.add_argument(
        '--base-port', type=int, help='Worker nodes will communicate with clients '
        'over ports. Each worker-client pair will use the part that is the base '
        'port + its id.',
        default=9900
    )
    scale_parser.add_argument(
        "--inference-mode", type=str, default="local", choices=["local", "remote"]
    )
    return scale_parser


def run(full_config_path, parameters):
    from abmarl import scale
    scale.run(full_config_path, parameters)
