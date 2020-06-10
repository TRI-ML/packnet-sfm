# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from packnet_sfm.utils.image import match_scales
from packnet_sfm.losses.loss_base import LossBase


class VelocityLoss(LossBase):
    """
    Velocity loss for pose translation.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred_pose, gt_pose_context, **kwargs):
        """
        Calculates velocity loss.

        Parameters
        ----------
        pred_pose : list of Pose
            Predicted pose transformation between origin and reference
        gt_pose_context : list of Pose
            Ground-truth pose transformation between origin and reference

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        pred_trans = [pose.mat[:, :3, -1].norm(dim=-1) for pose in pred_pose]
        gt_trans = [pose[:, :3, -1].norm(dim=-1) for pose in gt_pose_context]
        # Calculate velocity supervision loss
        loss = sum([(pred - gt).abs().mean()
                    for pred, gt in zip(pred_trans, gt_trans)]) / len(gt_trans)
        self.add_metric('velocity_loss', loss)
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }
