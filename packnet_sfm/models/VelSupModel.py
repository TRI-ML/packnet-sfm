# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.models.SelfSupModel import SelfSupModel
from packnet_sfm.losses.velocity_loss import VelocityLoss


class VelSupModel(SelfSupModel):
    """
    Self-supervised model with additional velocity supervision loss.

    Parameters
    ----------
    velocity_loss_weight : float
        Weight for velocity supervision
    kwargs : dict
        Extra parameters
    """
    def __init__(self, velocity_loss_weight=0.1, **kwargs):
        # Initializes SelfSupModel
        super().__init__(**kwargs)
        # Stores velocity supervision loss weight
        self._velocity_loss = VelocityLoss(**kwargs)
        self.velocity_loss_weight = velocity_loss_weight

        # GT pose is required
        self._train_requirements['gt_pose'] = True

    def forward(self, batch, return_logs=False, progress=0.0):
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
        output = super().forward(batch, return_logs, progress)
        if self.training:
            # Update self-supervised loss with velocity supervision
            velocity_loss = self._velocity_loss(output['poses'], batch['pose_context'])
            output['loss'] += self.velocity_loss_weight * velocity_loss['loss']
        return output
