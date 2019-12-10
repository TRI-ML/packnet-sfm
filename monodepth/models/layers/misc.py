# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
Various network layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv(nn.Module):
    """
    Base convolution 2D class
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                   stride=stride)
        self.normalize = torch.nn.GroupNorm(16, out_planes)
        p = kernel_size // 2
        self.p2d = (p, p, p, p)

    def forward(self, x):
        x = self.conv_base(F.pad(x, self.p2d))
        x = self.normalize(x)
        return F.elu(x, inplace=True)


class resconv_basic(nn.Module):
    """
    Base residual convolution class
    """
    def __init__(self, in_planes, out_planes, stride, dropout=None):
        super(resconv_basic, self).__init__()
        self.out_planes = out_planes
        self.stride = stride
        self.conv1 = conv(in_planes, out_planes, 3, stride)
        self.conv2 = conv(out_planes, out_planes, 3, 1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)
        self.normalize = torch.nn.GroupNorm(16, out_planes)

        if dropout:
            self.conv3 = nn.Sequential(self.conv3, nn.Dropout2d(dropout))

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        shortcut = self.conv3(x)
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


def resblock_basic(in_planes, out_planes, num_blocks, stride, dropout=None):
    """
    Base residual block class
    """
    layers = []
    layers.append(resconv_basic(in_planes, out_planes, stride, dropout=dropout))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(out_planes, out_planes, 1, dropout=dropout))
    return nn.Sequential(*layers)


class get_invdepth(nn.Module):
    """
    Inverse depth prediction module with a final sigmoid layer, scaled by
    1/max_depth. This allows the network to estimate depths of at least min_depth meters
    """
    def __init__(self, in_planes, out_planes=2, min_depth=0.5):
        super(get_invdepth, self).__init__()
        self.min_depth = min_depth
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.conv1(F.pad(x, (1, 1, 1, 1)))
        return F.sigmoid(x) / self.min_depth
