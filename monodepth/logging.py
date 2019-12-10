# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
Logging utilities
"""

from termcolor import colored


def printcolor_single(message, color="white"):
    """
    Prints output with a certain color
    """
    print(colored(message, color))


def printcolor(message, color="white"):
    """
    Prints output with a certain color
    """
    print(colored(message, color))


def print_error_metrics(error_names, error_values, screen_color, description):
    """
    Prints output with a certain color
    """
    errors_np = error_values.detach().cpu().numpy()
    error_string = ', '.join('{} : {:7.3f}'.format(name, error) for name, error in zip(error_names, errors_np))
    printcolor_single('{:>10} | Avg {}'.format(description.upper(), error_string), screen_color)
