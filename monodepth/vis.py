# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
Basic visualization utilities for depth estimation.
"""

import torch
from matplotlib.cm import get_cmap
import numpy as np


def prep_tensor_for_vis(x):
    """Prepare tensor for visualization
    If only has one channel, concatenate to produce 3 channels
    Clone, detach and pass to cpu before clamping between 0 and 1

    Parameters
    ----------
    x: torch.FloatTensor
        Tensor with image (CHW)

    Returns
    ----------
    torch.FloatTensor
        3HW detached tensor on cpu clamped between 0 and 1
    """
    if x.shape[0] == 1:
        x = torch.cat([x] * 3, 0)
    return torch.clamp(x.clone().detach().cpu(), 0., 1.)


def vis_inverse_depth(inv_depth, normalizer=None, percentile=95,
                      colormap='plasma', filter_zeros=False):
    """Visualize inverse depth with provided colormap.
    Parameters
    ----------
    inv_depth: np.ndarray
        Inverse depth to be visualized
    normalizer: float or None
        Normalize inverse depth by dividing by this factor
    percentile: float (default: 95)
        Use this percentile to normalize the inverse depth, if normalizer is
        not provided.
    colormap: str (default: plasma)
        Colormap used for visualizing the inverse depth
    Returns
    ----------
    vis: 3-channel float32 np.ndarray (HW3) with values between (0,1)
    """
    # Note: expects (H,W) shape
    cm = get_cmap(colormap)
    if normalizer is None:
        normalizer = np.percentile(
            inv_depth[inv_depth > 0] if filter_zeros else inv_depth, percentile)
    inv_depth /= (normalizer + 1e-6)
    return cm(np.clip(inv_depth, 0., 1.0))[:, :, :3]


