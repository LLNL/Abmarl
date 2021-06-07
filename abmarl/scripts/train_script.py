def create_parser(subparsers):
    """Parse the arguments for the train command.

    Returns
    -------
        parser : ArgumentParser
    """
    train_parser = subparsers.add_parser('train', help='Train MARL policies ')
    train_parser.add_argument(
        'configuration', type=str, help='Path to python config file. Include the .py extension.'
    )
    return train_parser


def run(full_config_path):
    from abmarl import train
    train.run(full_config_path)
