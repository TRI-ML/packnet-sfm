# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
Packing/unpacking functions
"""

import torch.nn as nn
from functools import partial
from monodepth.models.layers.misc import conv


def packing(x, r=2):
    """Takes a BCHW tensor and returns a B(rC)(H/r)(W/r) tensor,
    by concatenating neighbor spatial pixels as extra channels.
    It is the inverse of nn.PixelShuffle (if you apply both sequentially you should get the same tensor)
    Example r=2: A RGB image (C=3) becomes RRRRGGGGBBBB (C=12) and is downsampled to half its size
    """
    b, c, h, w = x.shape
    out_channel = c * (r ** 2)
    out_h, out_w = h // r, w // r
    x = x.contiguous().view(b, c, out_h, r, out_w, r)
    return x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)


class PackLayerConv2d(nn.Module):
    """Packing layer with 2d convolutions.
    Takes a BCHW tensor, packs it into B4CH/2W/2 and then convolves it to
    produce BCH/2W/2.
    """
    def __init__(self, in_channels, kernel_size, r=2):
        super().__init__()
        self.conv = conv(in_channels * (r ** 2), in_channels, kernel_size, 1)
        self.pack = partial(packing, r=r)

    def forward(self, x):
        x = self.pack(x)
        x = self.conv(x)
        return x


class UnpackLayerConv2d(nn.Module):
    """Unpacking layer with 2d convolutions.
    Takes a BCHW tensor, convolves it to produce B4CHW and then unpacks it to
    produce BC2H2W.
    """
    def __init__(self, in_channels, out_channels, kernel_size, r=2):
        super().__init__()
        self.conv = conv(in_channels, out_channels * (r ** 2), kernel_size, 1)
        self.unpack = nn.PixelShuffle(r)

    def forward(self, x):
        x = self.conv(x)
        x = self.unpack(x)
        return x


class PackLayerConv3d(nn.Module):
    """Packing layer with 3d convolutions.
    Takes a BCHW tensor, packs it into B4CH/2W/2 and then convolves it to
    produce BCH/2W/2.
    """
    def __init__(self, in_channels, kernel_size, r=2, d=8):
        super().__init__()
        self.conv = conv(in_channels * (r ** 2) * d, in_channels, kernel_size, 1)
        self.pack = partial(packing, r=r)
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        x = self.pack(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        x = self.conv(x)
        return x


class UnpackLayerConv3d(nn.Module):
    """Unpacking layer with 3d convolutions.
    Takes a BCHW tensor, convolves it to produce B4CHW and then unpacks it to
    produce BC2H2W.
    """
    def __init__(self, in_channels, out_channels, kernel_size, r=2, d=8):
        super().__init__()
        self.conv = conv(in_channels, out_channels * (r ** 2) // d, kernel_size, 1)
        self.unpack = nn.PixelShuffle(r)
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        x = self.unpack(x)
        return x