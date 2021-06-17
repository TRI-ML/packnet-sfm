# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch

from packnet_sfm.models.SelfSupModel import SfmModel, SelfSupModel
from packnet_sfm.losses.supervised_loss import SupervisedLoss
from packnet_sfm.models.model_utils import merge_outputs
from packnet_sfm.utils.depth import depth2inv


class SemiSupCompletionModel(SelfSupModel):
    """
    Semi-Supervised model for depth prediction and completion.

    Parameters
    ----------
    supervised_loss_weight : float
        Weight for the supervised loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_loss_weight=0.9, weight_rgbd=1.0, **kwargs):
        # Initializes SelfSupModel
        super().__init__(**kwargs)
        # If supervision weight is 0.0, use SelfSupModel directly
        assert 0. < supervised_loss_weight <= 1., "Model requires (0, 1] supervision"
        # Store weight and initializes supervised loss
        self.supervised_loss_weight = supervised_loss_weight
        self._supervised_loss = SupervisedLoss(**kwargs)

        # Pose network is only required if there is self-supervision
        if self.supervised_loss_weight == 1:
            self._network_requirements.remove('pose_net')
        # GT depth is only required if there is supervision
        if self.supervised_loss_weight > 0:
            self._train_requirements.append('gt_depth')

        self._input_keys = ['rgb', 'input_depth', 'intrinsics']

        self.weight_rgbd = weight_rgbd

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._supervised_loss.logs
        }

    def supervised_loss(self, inv_depths, gt_inv_depths,
                        return_logs=False, progress=0.0):
        """
        Calculates the supervised loss.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        gt_inv_depths : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth maps from the original image
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._supervised_loss(
            inv_depths, gt_inv_depths,
            return_logs=return_logs, progress=progress)

    def forward(self, batch, return_logs=False, progress=0.0, **kwargs):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and different metrics and predictions
            for logging and downstream usage.
        """
        if not self.training:
            # If not training, no need for self-supervised loss
            return SfmModel.forward(self, batch, return_logs=return_logs, **kwargs)
        else:
            if self.supervised_loss_weight == 1.:
                # If no self-supervision, no need to calculate loss
                self_sup_output = SfmModel.forward(self, batch, return_logs=return_logs, **kwargs)
                loss = torch.tensor([0.]).type_as(batch['rgb'])
            else:
                # Otherwise, calculate and weight self-supervised loss
                self_sup_output = SelfSupModel.forward(
                    self, batch, return_logs=return_logs, progress=progress, **kwargs)
                loss = (1.0 - self.supervised_loss_weight) * self_sup_output['loss']
            # Calculate and weight supervised loss
            sup_output = self.supervised_loss(
                self_sup_output['inv_depths'], depth2inv(batch['depth']),
                return_logs=return_logs, progress=progress)
            loss += self.supervised_loss_weight * sup_output['loss']
            if 'inv_depths_rgbd' in self_sup_output:
                sup_output2 = self.supervised_loss(
                    self_sup_output['inv_depths_rgbd'], depth2inv(batch['depth']),
                    return_logs=return_logs, progress=progress)
                loss += self.weight_rgbd * self.supervised_loss_weight * sup_output2['loss']
                if 'depth_loss' in self_sup_output:
                    loss += self_sup_output['depth_loss']
            # Merge and return outputs
            return {
                'loss': loss,
                **merge_outputs(self_sup_output, sup_output),
            }
