# Copyright 2020 Toyota Research Institute.  All rights reserved.

import importlib
import warnings
from collections import OrderedDict
from importlib.util import find_spec, module_from_spec

import torch

from monodepth.utils import get_network_version, same_shape
from monodepth.logging import printcolor


def load_class(filename, params, paths='monodepth.models', verbose=None):
    """Utility for fetching and initializing a class

    Parameters
    ----------
    filename: str
        File name. Assumes that it contains a class of the same name
    params: dict
        Parameters to be provided to the network
    paths: list (str)
        Paths from which the class will be loaded
    verbose: str
        Optional parameter to print the class name when loading

    Returns
    -------
    torch.nn.Module
        Initialized class
    """
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    if verbose:
        printcolor('{}: {}'.format(verbose, filename), 'green')
    for path in paths:
        name = '{}.{}'.format(path, filename)
        if importlib.util.find_spec(name):
            module = importlib.import_module(name)
            cls = getattr(module, filename)
            return cls(**params)
    raise ValueError('Unknown class {}'.format(filename))


def get_state_dict(state_dict, module_key='module'):
    updated_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(module_key + '.'):
            newk = k[7:]
            updated_state_dict[newk] = v
        else:
            updated_state_dict[k] = v
    return updated_state_dict


def load_state_dict(model, checkpoint, key='state_dict', module_key='module'):
    if key in checkpoint:
        state_dict = checkpoint[key]
        updated_state_dict = get_state_dict(state_dict, module_key=module_key)
        model.load_state_dict(updated_state_dict)
    else:
        model.load_state_dict(checkpoint)
        warnings.warn('Loading deprecated model weights.')
    return model


def load_net_from_checkpoint(model, path, starts_with):
    checkpoint = torch.load(path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        updated_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith(starts_with):
                newk = k[len(starts_with) + 1:]  # +1 to remove the . after the network name
                updated_state_dict[newk] = v
            # else:
            #     updated_state_dict[k] = v
        model.load_state_dict(updated_state_dict)
    else:
        model.load_state_dict(checkpoint)
        warnings.warn('Loading deprecated model weights.')
    return model

#
# def sample_to_cuda(data):
#     if isinstance(data, dict):
#         data_cuda = {}
#         for key in data.keys():
#             data_cuda[key] = sample_to_cuda(data[key])
#         return data_cuda
#     elif isinstance(data, list):
#         data_cuda = []
#         for key in data:
#             data_cuda.append(sample_to_cuda(key))
#         return data_cuda
#     else:
#         return data.to('cuda')


def monodepth_beta(name, out_planes=1, dropout=None,
                   bn=False, store_features=None):
    name, version = get_network_version(name)
    disp_paths = ['monodepth.models.networks',
                  'monodepth.legacy.models.networks']
    network = load_class(name, paths=disp_paths,
                         params={'out_planes': out_planes, 'dropout': dropout,
                                 'version': version, 'bn': bn, 'store_features': store_features},
                         verbose='DepthNet')
    return network
