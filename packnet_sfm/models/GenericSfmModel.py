# Copyright 2020 Toyota Research Institute.  All rights reserved.

import random
import torch.nn as nn
from packnet_sfm.geometry.pose import Pose
from packnet_sfm.utils.misc import make_list
from packnet_sfm.models.SfmModel import SfmModel
import torch.nn.functional as F

class GenericSfmModel(SfmModel):
    """
    Model class encapsulating a pose and depth networks.

    Parameters
    ----------
    depth_net : nn.Module
        Depth network to be used
    pose_net : nn.Module
        Pose network to be used
    rotation_mode : str
        Rotation mode for the pose network
    flip_lr_prob : float
        Probability of flipping when using the depth network
    upsample_depth_maps : bool
        True if depth map scales are upsampled to highest resolution
    kwargs : dict
        Extra parameters
    """

    def flip_model(self, model, image, flip):
        """
        Flip input image and flip output inverse depth map

        Parameters
        ----------
        model : nn.Module
            Module to be used
        image : torch.Tensor [B,3,H,W]
            Input image
        flip : bool
            True if the flip is happening

        Returns
        -------
        inv_depths : list of torch.Tensor [B,1,H,W]
            List of predicted inverse depth maps
        """
        if flip:
            return [flip_lr(inv_depth) for inv_depth in model(flip_lr(image))]
        else:
            return model(image)

    def interpolate_scales(self, images, shape=None, mode='bilinear', align_corners=False):
        """
        Interpolate list of images to the same shape

        Parameters
        ----------
        images : list of torch.Tensor [B,?,?,?]
            Images to be interpolated, with different resolutions
        shape : tuple (H, W)
            Output shape
        mode : str
            Interpolation mode
        align_corners : bool
            True if corners will be aligned after interpolation

        Returns
        -------
        images : list of torch.Tensor [B,?,H,W]
            Interpolated images, with the same resolution
        """
        # If no shape is provided, interpolate to highest resolution
        if shape is None:
            shape = images[0].shape
        # Take last two dimensions as shape
        if len(shape) > 2:
            shape = shape[-2:]
        # Interpolate all images
        return [F.interpolate(image, shape, mode=mode,
                                  align_corners=align_corners) for image in images]

    def compute_depth_net(self, image):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        inv_depths, raysurf = self.flip_model(self.depth_net, image, False)
        inv_depths = make_list(inv_depths)
        # If upsampling depth maps
        if self.upsample_depth_maps:
            inv_depths = self.interpolate_scales(
                inv_depths, mode='nearest', align_corners=None)
        # Return inverse depth maps
        return inv_depths, raysurf

    def forward(self, batch, return_logs=False):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored

        Returns
        -------
        output : dict
            Dictionary containing predicted inverse depth maps and poses
        """
        #print(logs)
        # Generate inverse depth predictions
        inv_depths, raysurf = self.compute_depth_net(batch['rgb'])
        # Generate pose predictions if available
        pose = None
        if 'rgb_context' in batch and self.pose_net is not None:
            pose = self.compute_poses(batch['rgb'],
            #pose = self.compute_pose_net(batch['rgb'],
                                      batch['rgb_context'])
        # Return output dictionary
        return {
            'inv_depths': inv_depths,
            'poses': pose,
            'ray_surface': raysurf
        }
