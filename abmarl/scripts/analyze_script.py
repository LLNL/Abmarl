def create_parser(subparsers):
    analyze_parser = subparsers.add_parser('analyze', help='Analyze MARL policies')
    analyze_parser.add_argument(
        'configuration', type=str, help='Path to saved policy directory.'
    )
    analyze_parser.add_argument(
        'subscript', type=str, help='Path to subscript to run.'
    )
    analyze_parser.add_argument(
        '-c', '--checkpoint', type=int,
        help='Specify which checkpoint to load. Default is the last timestep in the directory.'
    )
    analyze_parser.add_argument('--seed', type=int, help='Seed for reproducibility.')
    return analyze_parser


def run(full_trained_directory, full_subscript, parameters):
    from abmarl import stage
    stage.run_analysis(full_trained_directory, full_subscript, parameters)
