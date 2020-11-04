
def create_parser(subparsers):
    play_parser = subparsers.add_parser('play', help='Play MARL policies')
    play_parser.add_argument('configuration', type=str, \
        help='Path to saved policy directory.')
    play_parser.add_argument('-c','--checkpoint', type=int, \
        help='Specify which checkpoint to load. Default is the last timestep in the directory.')
    play_parser.add_argument('-n', '--episodes', type=int, default=1, \
        help='The number of episodes to run. Default 1.')
    play_parser.add_argument('--record', action='store_true', \
        help='Record a video of the agent interacting in the environment.')
    play_parser.add_argument('--frame-delay', type=int, help='The number of milliseconds ' + \
        'to delay between each frame in the animation.', default=200)
    play_parser.add_argument('--no-explore', action='store_false', help='Turn off' + \
        'exploration in the action policy.')
    return play_parser

def run(full_trained_directory, parameters):
    from admiral import play
    play.run(full_trained_directory, parameters)