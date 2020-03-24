# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
PackNet model with 3d convolutions.
"""

import torch
import torch.nn as nn
from monodepth.models.layers.misc import conv, resblock_basic, get_invdepth
from monodepth.models.layers.packnet import PackLayerConv3d, UnpackLayerConv3d


class PackNet01(nn.Module):
    def __init__(self, in_planes=3, out_planes=2, dropout=None, version=None, bn=False,
                 store_features=None):
        super().__init__()
        assert not bn, 'Only GroupNorm is supported'
        self.super_resolution = int(version[0]) > 1
        out_planes = out_planes * int(version[0])
        self.version = version[1:]
        self.store_features = store_features
        self.features = {}

        # Hyper-parameters
        ni, no = 64, out_planes
        n1, n2, n3, n4, n5 = 64, 64, 128, 256, 512
        num_blocks = [2, 2, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]
        self.pre_calc = conv(in_planes, ni, 5, 1)

        # Version A (Concatenated features):
        # Concatenate upconv features, skip features and up-sampled disparities
        if self.version == 'A':
            n1o, n1i = n1, n1 + ni + no
            n2o, n2i = n2, n2 + n1 + no
            n3o, n3i = n3, n3 + n2 + no
            n4o, n4i = n4, n4 + n3
            n5o, n5i = n5, n5 + n4
        # Version B (Additive features):
        # Add upconv features and skip features, and concatenate the result
        # with the upsampled disparities
        elif self.version == 'B':
            n1o, n1i = n1, n1 + no
            n2o, n2i = n2, n2 + no
            n3o, n3i = n3//2, n3//2 + no
            n4o, n4i = n4//2, n4//2
            n5o, n5i = n5//2, n5//2
        else:
            raise ValueError('Unknown PackNet version {}'.format(version))

        # Encoder
        self.pack1 = PackLayerConv3d(n1, pack_kernel[0])
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1])
        self.pack3 = PackLayerConv3d(n3, pack_kernel[2])
        self.pack4 = PackLayerConv3d(n4, pack_kernel[3])
        self.pack5 = PackLayerConv3d(n5, pack_kernel[4])

        self.conv1 = conv(ni, n1, 7, 1)
        self.conv2 = resblock_basic(n1, n2, num_blocks[0], 1, dropout=dropout)
        self.conv3 = resblock_basic(n2, n3, num_blocks[1], 1, dropout=dropout)
        self.conv4 = resblock_basic(n3, n4, num_blocks[2], 1, dropout=dropout)
        self.conv5 = resblock_basic(n4, n5, num_blocks[3], 1, dropout=dropout)

        # Decoder
        self.unpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0])
        self.unpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1])
        self.unpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2])
        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3])
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4])

        self.iconv5 = conv(n5i, n5, iconv_kernel[0], 1)
        self.iconv4 = conv(n4i, n4, iconv_kernel[1], 1)
        self.iconv3 = conv(n3i, n3, iconv_kernel[2], 1)
        self.iconv2 = conv(n2i, n2, iconv_kernel[3], 1)
        self.iconv1 = conv(n1i, n1, iconv_kernel[4], 1)

        # Depth Layers
        self.unpack_disps = nn.PixelShuffle(2)
        self.unpack_disp4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.disp4_layer = get_invdepth(n4, out_planes=out_planes)
        self.disp3_layer = get_invdepth(n3, out_planes=out_planes)
        self.disp2_layer = get_invdepth(n2, out_planes=out_planes)
        self.disp1_layer = get_invdepth(n1, out_planes=out_planes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.pre_calc(x)

        # Encoder
        x1 = self.conv1(x)
        x1p = self.pack1(x1)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)
        x3 = self.conv3(x2p)
        x3p = self.pack3(x3)
        x4 = self.conv4(x3p)
        x4p = self.pack4(x4)
        x5 = self.conv5(x4p)
        x5p = self.pack5(x5)

        # Skips
        skip1 = x
        skip2 = x1p
        skip3 = x2p
        skip4 = x3p
        skip5 = x4p

        # Decoder
        unpack5 = self.unpack5(x5p)
        if self.version == 'A':
            concat5 = torch.cat((unpack5, skip5), 1)
        else:
            concat5 = unpack5 + skip5
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        if self.version == 'A':
            concat4 = torch.cat((unpack4, skip4), 1)
        else:
            concat4 = unpack4 + skip4
        iconv4 = self.iconv4(concat4)
        disp4 = self.disp4_layer(iconv4)
        udisp4 = self.unpack_disp4(disp4)

        unpack3 = self.unpack3(iconv4)
        if self.version == 'A':
            concat3 = torch.cat((unpack3, skip3, udisp4), 1)
        else:
            concat3 = torch.cat((unpack3 + skip3, udisp4), 1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3_layer(iconv3)
        udisp3 = self.unpack_disp3(disp3)

        unpack2 = self.unpack2(iconv3)
        if self.version == 'A':
            concat2 = torch.cat((unpack2, skip2, udisp3), 1)
        else:
            concat2 = torch.cat((unpack2 + skip2, udisp3), 1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2_layer(iconv2)
        udisp2 = self.unpack_disp2(disp2)

        unpack1 = self.unpack1(iconv2)
        if self.version == 'A':
            concat1 = torch.cat((unpack1, skip1, udisp2), 1)
        else:
            concat1 = torch.cat((unpack1 +  skip1, udisp2), 1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1_layer(iconv1)

        if self.store_features is not None:
            local_vars = locals()
            self.features = {key: local_vars[key] for key in self.store_features}

        if self.training:
            # For SR option, we super-resolve images:
            # (B,r^2*C,H,W) images -> B,C,r*H,r*W
            if self.super_resolution:
                disp1 = self.unpack_disps(disp1)
                disp2 = self.unpack_disps(disp2)
                disp3 = self.unpack_disps(disp3)
                disp4 = self.unpack_disps(disp4)
            return disp1, disp2, disp3, disp4
        else:
            # For SR option, we super-resolve images:
            # (B,r^2*C,H,W) images -> B,C,r*H,r*W
            if self.super_resolution:
                disp1 = self.unpack_disps(disp1)
            return disp1

