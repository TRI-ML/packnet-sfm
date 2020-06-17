# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
from termcolor import colored
from functools import partial

from packnet_sfm.utils.horovod import on_rank_0


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def prepare_dataset_prefix(config, dataset_idx):
    """
    Concatenates dataset path and split for metrics logging

    Parameters
    ----------
    config : CfgNode
        Dataset configuration
    dataset_idx : int
        Dataset index for multiple datasets

    Returns
    -------
    prefix : str
        Dataset prefix for metrics logging
    """
    # Path is always available
    prefix = '{}'.format(os.path.splitext(config.path[dataset_idx].split('/')[-1])[0])
    # If split is available and does not contain { character
    if config.split[dataset_idx] != '' and '{' not in config.split[dataset_idx]:
        prefix += '-{}'.format(os.path.splitext(os.path.basename(config.split[dataset_idx]))[0])
    # If depth type is available
    if config.depth_type[dataset_idx] != '':
        prefix += '-{}'.format(config.depth_type[dataset_idx])
    # If we are using specific cameras
    if len(config.cameras[dataset_idx]) == 1:  # only allows single cameras
        prefix += '-{}'.format(config.cameras[dataset_idx][0])
    # Return full prefix
    return prefix


def s3_url(config):
    """
    Generate the s3 url where the models will be saved

    Parameters
    ----------
    config : CfgNode
        Model configuration

    Returns
    -------
    url : str
        String containing the URL pointing to the s3 bucket
    """
    return 'https://s3.console.aws.amazon.com/s3/buckets/{}/{}'.format(
        config.checkpoint.s3_path[5:], config.name)


@on_rank_0
def print_config(config, color=('blue', 'red', 'cyan'), attrs=('bold', 'dark')):
    """
    Prints header for model configuration

    Parameters
    ----------
    config : CfgNode
        Model configuration
    color : list of str
        Color pallete for the header
    attrs :
        Colored string attributes
    """
    # Recursive print function
    def print_recursive(rec_args, n=2, l=0):
        if l == 0:
            print(pcolor('config:', color[1], attrs=attrs))
        for key, val in rec_args.items():
            if isinstance(val, dict):
                print(pcolor('{} {}:'.format('-' * n, key), color[1], attrs=attrs))
                print_recursive(val, n + 2, l + 1)
            else:
                print('{}: {}'.format(pcolor('{} {}'.format('-' * n, key), color[2]), val))

    # Color partial functions
    pcolor1 = partial(pcolor, color='blue', attrs=['bold', 'dark'])
    pcolor2 = partial(pcolor, color='blue', attrs=['bold'])
    # Config and name
    line = pcolor1('#' * 120)
    path = pcolor1('### Config: ') + \
           pcolor2('{}'.format(config.default.replace('/', '.'))) + \
           pcolor1(' -> ') + \
           pcolor2('{}'.format(config.config.replace('/', '.')))
    name = pcolor1('### Name: ') + \
           pcolor2('{}'.format(config.name))
    # Add wandb link if available
    if not config.wandb.dry_run:
        name += pcolor1(' -> ') + \
                pcolor2('{}'.format(config.wandb.url))
    # Add s3 link if available
    if config.checkpoint.s3_path is not '':
        name += pcolor1('\n### s3:') + \
                pcolor2(' {}'.format(config.checkpoint.s3_url))
    # Create header string
    header = '%s\n%s\n%s\n%s' % (line, path, name, line)

    # Print header, config and header again
    print()
    print(header)
    print_recursive(config)
    print(header)
    print()


class AvgMeter:
    """Average meter for logging"""
    def __init__(self, n_max=100):
        """
        Initializes a AvgMeter object.

        Parameters
        ----------
        n_max : int
            Number of steps to average over
        """
        self.n_max = n_max
        self.values = []

    def __call__(self, value):
        """Appends new value and returns average"""
        self.values.append(value)
        if len(self.values) > self.n_max:
            self.values.pop(0)
        return self.get()

    def get(self):
        """Get current average"""
        return sum(self.values) / len(self.values)

    def reset(self):
        """Reset meter"""
        self.values.clear()

    def get_and_reset(self):
        """Gets current average and resets"""
        average = self.get()
        self.reset()
        return average
