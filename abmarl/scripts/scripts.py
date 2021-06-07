#!/usr/bin/env python

import argparse
import os

from abmarl.scripts import train_script as train
from abmarl.scripts import analyze_script as analyze
from abmarl.scripts import visualize_script as visualize
from abmarl.scripts import make_runnable_script as runnable

EXAMPLE_USAGE = """
Example usage for training:
    abmarl train my_experiment.py

Example usage for analysis:
    abmarl analyze my_experiment_directory/ my_analysis_script.py

Example usage for visualizing:
    abmarl visualize my_experiment_directory/ --some-args

Example usage for converting to runnable script:
    abmarl make-runnable my_experiment.py --some-args
"""


def cli():
    parser = argparse.ArgumentParser(
        prog='abmarl',
        description="Train, analyze, and visualize MARL policies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_USAGE
    )
    subparsers = parser.add_subparsers(dest='command')

    train.create_parser(subparsers)
    analyze.create_parser(subparsers)
    visualize.create_parser(subparsers)
    runnable.create_parser(subparsers)
    parameters = parser.parse_args()
    path_config = os.path.join(os.getcwd(), parameters.configuration)

    if parameters.command == 'train':
        train.run(path_config)
    elif parameters.command == 'analyze':
        full_path_subscript = os.path.join(os.getcwd(), parameters.subscript)
        analyze.run(path_config, full_path_subscript, parameters)
    elif parameters.command == 'visualize':
        visualize.run(path_config, parameters)
    elif parameters.command == 'make-runnable':
        runnable.run(path_config, parameters)
    else:
        parser.print_help()
