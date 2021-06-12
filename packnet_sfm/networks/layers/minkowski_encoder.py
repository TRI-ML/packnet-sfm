# Copyright 2020 Toyota Research Institute.  All rights reserved.

import MinkowskiEngine as ME
import torch.nn as nn

from packnet_sfm.networks.layers.minkowski import \
    sparsify_depth, densify_features, densify_add_features_unc, map_add_features


class MinkConv2D(nn.Module):
    """
    Minkowski Convolutional Block

    Parameters
    ----------
    in_planes : number of input channels
    out_planes : number of output channels
    kernel_size : convolutional kernel size
    stride : convolutional stride
    with_uncertainty : with uncertainty or now
    add_rgb : add RGB information as channels
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride,
                 with_uncertainty=False, add_rgb=False):
        super().__init__()
        self.layer3 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_planes, out_planes * 2, kernel_size=kernel_size, stride=1, dimension=2),
            ME.MinkowskiBatchNorm(out_planes * 2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_planes * 2, out_planes * 2, kernel_size=kernel_size, stride=1, dimension=2),
            ME.MinkowskiBatchNorm(out_planes * 2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_planes * 2, out_planes, kernel_size=kernel_size, stride=1, dimension=2),
        )

        self.layer2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_planes, out_planes * 2, kernel_size=kernel_size, stride=1, dimension=2),
            ME.MinkowskiBatchNorm(out_planes * 2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_planes * 2, out_planes, kernel_size=kernel_size, stride=1, dimension=2),
        )

        self.layer1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_planes, out_planes, kernel_size=kernel_size, stride=1, dimension=2),
        )

        self.layer_final = nn.Sequential(
            ME.MinkowskiBatchNorm(out_planes),
            ME.MinkowskiReLU(inplace=True)
        )
        self.pool = None if stride == 1 else ME.MinkowskiMaxPooling(3, stride, dimension=2)

        self.add_rgb = add_rgb
        self.with_uncertainty = with_uncertainty
        if with_uncertainty:
            self.unc_layer = nn.Sequential(
                ME.MinkowskiConvolution(
                    out_planes, 1, kernel_size=3, stride=1, dimension=2),
                ME.MinkowskiSigmoid()
            )

    def forward(self, x):
        """
        Processes sparse information

        Parameters
        ----------
        x : Sparse tensor

        Returns
        -------
        Processed tensor
        """
        if self.pool is not None:
            x = self.pool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        return None, self.layer_final(x1 + x2 + x3)


class MinkowskiEncoder(nn.Module):
    """
    Depth completion Minkowski Encoder

    Parameters
    ----------
    channels : number of channels
    with_uncertainty : with uncertainty or not
    add_rgb : add RGB information to depth features or not
    """
    def __init__(self, channels, with_uncertainty=False, add_rgb=False):
        super().__init__()
        self.mconvs = nn.ModuleList()
        kernel_sizes = [5, 5] + [3] * (len(channels) - 1)
        self.mconvs.append(
            MinkConv2D(1, channels[0], kernel_sizes[0], 2,
                       with_uncertainty=with_uncertainty))
        for i in range(0, len(channels) - 1):
            self.mconvs.append(
                MinkConv2D(channels[i], channels[i+1], kernel_sizes[i+1], 2,
                           with_uncertainty=with_uncertainty))
        self.d = self.n = self.shape = 0
        self.with_uncertainty = with_uncertainty
        self.add_rgb = add_rgb

    def prep(self, d):
        self.d = sparsify_depth(d)
        self.shape = d.shape
        self.n = 0

    def forward(self, x=None):

        unc, self.d = self.mconvs[self.n](self.d)
        self.n += 1

        if self.with_uncertainty:
            out = densify_add_features_unc(x, unc * self.d, unc, self.shape)
        else:
            out = densify_features(self.d, self.shape)

        if self.add_rgb:
            self.d = map_add_features(x, self.d)

        return out
