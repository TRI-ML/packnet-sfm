# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
Various utilities
"""

import torch
import numpy as np
from monodepth.functional.image import fliplr


def same_shape(v1, v2):
    """
    Check if two variables have the same shape
    :param v1: variable 1 (np.array or torch.tensor)
    :param v2: variable 2 (np.array or torch.tensor)
    :return: bool (True if the same shape, False if otherwise)
    """
    if len(v1.shape) != len(v2.shape):
        return False
    for i in range(len(v1.shape)):
        if v1.shape[i] != v2.shape[i]:
            return False
    return True


def get_network_version(name):
    """
    Returns the network name and version (everything after _)
    Useful to customize the network on the fly
    """
    idx = name.rfind('_')
    if idx < 0:
        return name, None
    else:
        return name[:idx], name[idx + 1:]


def fuse_disparity(ldisp, ldisp_hat, method='mean'):
    """
    Fuses two predicted depths into a single one
    """
    if method == 'mean':
        ldisp_fused = 0.5 * (ldisp + ldisp_hat)
    elif method == 'max':
        ldisp_fused = torch.max(ldisp, ldisp_hat)
    elif method == 'min':
        ldisp_fused = torch.min(ldisp, ldisp_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))
    return ldisp_fused


def post_process_disparity_with_border_ramps(ldisp, ldisp_flipped, method='mean'):
    """
    # Create two ramps on both sides of the image for fusing the left
    # (5% of the width), and flipped left disparities at the edges.
    """
    B, C, H, W = ldisp.shape
    ldisp_hat = fliplr(ldisp_flipped)
    ldisp_fused = fuse_disparity(ldisp, ldisp_hat, method=method)

    xs = torch.linspace(0,1,W, device=ldisp.device, dtype=ldisp.dtype).repeat(B,C,H,1)
    lmask = 1.0 - torch.clamp(20 * (xs - 0.05), 0, 1)
    lmask_hat = fliplr(lmask)
    return lmask_hat * ldisp + lmask * ldisp_hat + \
        (1.0 - lmask - lmask_hat) * ldisp_fused
