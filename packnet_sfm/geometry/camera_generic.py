# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import lru_cache
import torch
import torch.nn as nn

import numpy as np

from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera_utils import scale_intrinsics
from packnet_sfm.utils.image import image_grid
import torch.nn.functional as F

########################################################################################################################


class GenericCamera(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for a pinhole model.
    """

    def __init__(self, R, Tcw=None):
        """
        Initializes the Camera class

        Parameters
        ----------
        R : torch.Tensor [B, 3, H, W]
            Camera ray surface
        Tcw : Pose
            Camera -> World pose transformation
        """
        super().__init__()
        self.ray_surface = R
        self.Tcw = Pose.identity(1) if Tcw is None else Tcw

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.ray_surface = self.ray_surface.to(*args, **kwargs)
        self.Tcw = self.Tcw.to(*args, **kwargs)
        return self

########################################################################################################################

    @property
    @lru_cache()
    def Twc(self):
        """World -> Camera pose transformation (inverse of Tcw)"""
        return self.Tcw.inverse()

########################################################################################################################

    def reconstruct(self, depth, frame='w'):
        """
        Reconstructs pixel-wise 3D points from a depth map: P(x, y) = s(x, y) + d(x, y) * r(x, y)

        Parameters
        ----------
        depth : torch.Tensor [B,1,H,W]
            Depth map for the camera
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.tensor [B,3,H,W]
            Pixel-wise 3D points
        """

        B, C, H, W = depth.shape
        assert C == 1

        Xc = self.ray_surface * depth[0].unsqueeze(0)

        # If in camera frame of reference
        if frame == 'c':
            return Xc
        # If in world frame of reference
        elif frame == 'w':
            return self.Twc @ Xc
        # If none of the above
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

    def project(self, X, progress, downsample=True, frame='c'):
        """
        Projects 3D points onto the image plane. Approximated by softmax (see paper).

        Parameters
        ----------
        X : torch.Tensor [B,3,H,W]
            3D points to be projected
        progress : np.double 
            training progress
        downsample : True 
            whether to downsample projection tensor
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.Tensor [B,H,W,2]
            2D projected points that are within the image boundaries
        """

        B, C, H, W = X.shape
        assert C == 3

        ray_surface = self.ray_surface

        # hyperparameters for projection patching
        patch_side = 20
        K = (2*patch_side+1)**2

        # hyperparams for annealing
        min_temp = 1e-8
        start_temp = 0.0001
        constant = 0.1

        if frame == 'w':
            X = self.Tcw @ X

        def _make_grid(xi, xf, yi, yf, r=1., final=False):
            if final:
                xf += r
                yf += r
            x, y = torch.meshgrid([
                torch.arange(xi, xf, r),
                torch.arange(yi, yf, r)])
            return torch.stack([x, y], 2).long().cuda()

        def _get_patch_coords(h, w, kh, kw):

            grid = _make_grid(0, h, 0, w, final=False)
            patch = _make_grid(-kh, kh, -kw, kw, final=True)
            kh, kw = 2 * kh + 1, 2 * kw + 1
            grid = grid.reshape(-1, 2).unsqueeze(1).repeat([1, kh * kw, 1])
            patch = patch.reshape(-1, 2).unsqueeze(0).repeat([h * w, 1, 1])
            grid += patch

            ox = grid[:, 0, 0] < 0
            grid[ox, :, 0] -= grid[ox, 0, 0].unsqueeze(1)
            oy = grid[:, 0, 1] < 0
            grid[oy, :, 1] -= grid[oy, 0, 1].unsqueeze(1)
            ox = grid[:, -1, 0] > h - 1
            grid[ox, :, 0] -= grid[ox, -1, 0].unsqueeze(1) - (h - 1)
            oy = grid[:, -1, 1] > w - 1
            grid[oy, :, 1] -= grid[oy, -1, 1].unsqueeze(1) - (w - 1)
            return grid

        def _get_value_coords(x, coords):

            b, c, h, w = x.shape
            n, k, _ = coords.shape
            coords = coords.reshape(-1, 2)
            out = x[:, :, coords[:, 0], coords[:, 1]]
            out = out.reshape(b, c, h, w, k)
            return out

        if downsample:
            H, W = int(H/2.), int(W/2.)
            ray_surface = F.interpolate(
                ray_surface, mode='bilinear', scale_factor=0.5, align_corners=True)

        # patch-based projection
        patch_coordinates = _get_patch_coords(
            H, W, patch_side, patch_side).cuda()
        ray_surf_patch = _get_value_coords(
            ray_surface, patch_coordinates).cuda()

        direction = X

        if downsample:
            direction = F.interpolate(
                direction, mode='bilinear', scale_factor=0.5, align_corners=True)

        direction_norm = direction.view(B, 3, H*W)
        direction_norm = direction_norm / \
            torch.norm(direction_norm, dim=1, keepdim=True)
        direction_view = direction_norm.view(B, 3, H*W).squeeze()

        patch_ray_view = ray_surf_patch.squeeze().view(3, H*W, K)

        ray_logits = torch.bmm(direction_view.permute(1, 0).unsqueeze(
            1), patch_ray_view.permute(1, 0, 2)).squeeze()

        # Softmax with temperature
        temperature = np.maximum(
            min_temp, start_temp/np.exp(constant * progress))
        image_coords_softmax = torch.softmax(ray_logits / temperature, -1)

        image_coords = torch.bmm(
            image_coords_softmax.unsqueeze(1), patch_coordinates.float())
        image_coords = image_coords.squeeze().view(H, W, 2)

        Xnorm = 2 * image_coords[:, :, 0] / (H - 1) - 1.
        Ynorm = 2 * image_coords[:, :, 1] / (W - 1) - 1.

        if downsample:
            Xnorm = F.interpolate(Xnorm.unsqueeze(0).unsqueeze(
                0), mode='bilinear', scale_factor=2.0, align_corners=True).squeeze()
            Ynorm = F.interpolate(Ynorm.unsqueeze(0).unsqueeze(
                0), mode='bilinear', scale_factor=2.0, align_corners=True).squeeze()
            H = H * 2
            W = W * 2

        return torch.stack([Ynorm, Xnorm], dim=-1).view(B, H, W, 2)
