# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
Image utilities
"""

import torch
import torch.nn.functional as F


def scale_image(img, nh, nw, mode='bilinear', align_corners=True):
    """
    Scale image to desired height and width.

    Parameters
    ----------
    img: torch.FloatTensor (BHW or B3HW)
        Image to be scaled
    nh: int
        Desired image height
    nw: int
        Desired image width
    mode: str
        Interpolation mode for image resizing
    align_corners: bool
        Interpolation argument for upsampling

    Returns
    ----------
    torch.FloatTensor(s)
        Source `img` resized to (nh,nw)
    """
    B, _, H, W = img.shape
    if nh == H and nw == W:
        return img
    return F.interpolate(img, size=[nh, nw],
                         mode=mode, align_corners=align_corners)


def get_resized_depth(inv_depth, shape, mode='bilinear'):
    """
    Converts inverse depths to resized depths

    Parameters
    ----------
    inv_depth: torch.FloatTensor (BHW or B1HW)
        Predicted inverse depths
    shape: (height, width)
        Resized shape
    mode: str
        Interpolation method

    Returns
    -------
    torch.FloatTensor (BHW or B1HW)
        Resized predicted depths
    """
    return scale_image(1. / inv_depth.clamp(min=1e-6), shape[-2], shape[-1],
                       mode=mode, align_corners=True if mode == 'bilinear' else None)


def fliplr(img):
    """
    Flip image horizontally in a differentiable manner.

    Parameters
    ----------
    img: Torch.Tensor
        Image batch (BCHW) to be horizontally flipped along the last dimension.

    Returns
    -------
    torch.Tensor:
        Horizontally flipped image (BCHW).
    """
    assert img.dim() == 4
    return torch.flip(img, [3])
