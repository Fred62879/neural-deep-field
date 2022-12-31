
import argparse
import configargparse

from wisp.parsers.parse_yaml import parse_input_config


def parse_args():
    ''' Parse all command arguments and generate all needed ones. '''
    config, args_str = parse_input_config()
    args = argparse.Namespace(**config)
    return args, args_str
