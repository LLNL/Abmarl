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
    from abmarl.stage import analyze
    from abmarl.tools import utils as adu
    params = adu.find_params_from_output_dir(full_trained_directory)
    analysis_func = adu.custom_import_module(full_subscript).run
    import ray
    ray.init()
    analyze(params, full_trained_directory, analysis_func, **vars(parameters))
    ray.shutdown()
