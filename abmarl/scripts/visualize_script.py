def create_parser(subparsers):
    visualize_parser = subparsers.add_parser('visualize', help='Visualize MARL policies')
    visualize_parser.add_argument(
        'configuration', type=str, help='Path to saved policy directory.'
    )
    visualize_parser.add_argument(
        '-c', '--checkpoint', type=int,
        help='Specify which checkpoint to load. Default is the last timestep in the directory.'
    )
    visualize_parser.add_argument(
        '-n', '--episodes', type=int, default=1, help='The number of episodes to run. Default 1.'
    )
    visualize_parser.add_argument(
        '-s', '--steps-per-episode', type=int, default=200,
        help='The maximum number of steps to take per epsiode. Default 200.'
    )
    visualize_parser.add_argument(
        '--record', action='store_true',
        help='Record a video of the agent(s) interacting in the simulation and display it live.'
    )
    visualize_parser.add_argument(
        '--record-only', action='store_true',
        help='Only record a video of the agent(s) interacting in the simulation. No live display.'
    )
    visualize_parser.add_argument(
        '--frame-delay', type=int,
        help='The number of milliseconds to delay between each frame in the animation.',
        default=200
    )
    visualize_parser.add_argument(
        '--explore', action='store_true', help='Turn on exploration in the action policy.'
    )
    visualize_parser.add_argument('--seed', type=int, help='Seed for reproducibility.')
    return visualize_parser


def run(full_trained_directory, parameters):
    from abmarl.tools import utils as adu
    from abmarl.stage import visualize
    params = adu.find_params_from_output_dir(full_trained_directory)
    import ray
    ray.init()
    visualize(params, full_trained_directory, **vars(parameters))
    ray.shutdown()
