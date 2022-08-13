# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from packnet_sfm.utils.image import match_scales
from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling

from packnet_sfm.utils.depth import inv2depth
from packnet_sfm.utils.image import image_grid 
########################################################################################################################

class ReprojectedDistanceLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def warp(self, depth, K, Kinv, T):
    """
    Warps pixels from one image plane to another one.
    
    Parameters
    ----------
    depth : torch.Tensor [B,1,H,W]
        Depth map for the camera
    
    T : torch.Tensor 
      Relative pose

    K : torch.Tensor
      Intrinsics matrix

    Kinv : torch.Tensor
      Inverse of the intrinsics matrix

    Returns
    -------
    points : torch.Tensor [B,H,W,2]
            2D projected points that are within the image boundaries
    """

    B, _, H, W = depth.shape

    # Create flat index grid
    grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False)  # [B,3,H,W]
    flat_grid = grid.view(B, 3, -1)  # [B,3,HW]

    # Estimate the outward rays in the camera frame
    xnorm = (self.Kinv.bmm(flat_grid)).view(B, 3, H, W)
    # Scale rays to metric depth
    Xc = xnorm * depth

    # Project the 3D point
    Xs = T @ Xc
    Xs = self.K.bmm(Xs.view(B, 3, -1))
    # Normalize points
    X = Xs[:, 0]
    Y = Xs[:, 1]
    Z = Xs[:, 2].clamp(min=1e-5)
    Xnorm = 2 * (X / Z) / (W - 1) - 1.
    Ynorm = 2 * (Y / Z) / (H - 1) - 1.

    return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)

  def forward(self, inv_depth_gt, inv_depth_est, K, Kinv, T_ts):
    """
    Calculates the reprojected distance loss.

    Parameters
    ----------
    inv_depth_est : torch.Tensor [B,1,H,W]
        Predicted inverse depth map
    inv_depth_gt : torch.Tensor [B,1,H,W]
        Ground-truth inverse depth map
    K : torch.Tensor 
      Intrinsic parameter

    Kinv : torch.Tensor 
      Inverse of the intrinsics matrix

    T_ts : torch.Tensor
      Relative pose transformation from the source image camera frame to the target
      image camera frame
      
    
    Returns
    -------
    loss : torch.Tensor [1]
        reprojected distance loss
    """

    depth_gt = inv2depth(inv_depth_gt)
    depth_est = inv2depth(inv_depth_est)

    # Warp pixels from target frame to source frame
    # Convert pose from source-to-target into target-to-source
    T_st = torch.linalg.inv(T_ts)
    p_gt = self.warp(depth_gt, K, Kinv, T_st)
    p_est = self.warp(depth_est, K, Kinv, T_st)

    loss_tmp = (p_est - p_gt).abs()
    return loss_tmp.mean()



class BerHuLoss(nn.Module):
    """Class implementing the BerHu loss."""
    def __init__(self, threshold=0.2):
        """
        Initializes the BerHuLoss class.

        Parameters
        ----------
        threshold : float
            Mask parameter
        """
        super().__init__()
        self.threshold = threshold
    def forward(self, pred, gt):
        """
        Calculates the BerHu loss.

        Parameters
        ----------
        pred : torch.Tensor [B,1,H,W]
            Predicted inverse depth map
        gt : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth map

        Returns
        -------
        loss : torch.Tensor [1]
            BerHu loss
        """
        huber_c = torch.max(pred - gt)
        huber_c = self.threshold * huber_c
        diff = (pred - gt).abs()

        # Remove
        # mask = (gt > 0).detach()
        # diff = gt - pred
        # diff = diff[mask]
        # diff = diff.abs()

        huber_mask = (diff > huber_c).detach()
        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2
        return torch.cat((diff, diff2)).mean()

class SilogLoss(nn.Module):
    def __init__(self, ratio=10, ratio2=0.85):
        super().__init__()
        self.ratio = ratio
        self.ratio2 = ratio2

    def forward(self, pred, gt):
        log_diff = torch.log(pred * self.ratio) - \
                   torch.log(gt * self.ratio)
        silog1 = torch.mean(log_diff ** 2)
        silog2 = self.ratio2 * (log_diff.mean() ** 2)
        silog_loss = torch.sqrt(silog1 - silog2) * self.ratio
        return silog_loss

########################################################################################################################

def get_loss_func(supervised_method):
    """Determines the supervised loss to be used, given the supervised method."""
    if supervised_method.endswith('l1'):
        return nn.L1Loss()
    elif supervised_method.endswith('mse'):
        return nn.MSELoss()
    elif supervised_method.endswith('berhu'):
        return BerHuLoss()
    elif supervised_method.endswith('silog'):
        return SilogLoss()
    elif supervised_method.endswith('abs_rel'):
        return lambda x, y: torch.mean(torch.abs(x - y) / x)
    else:
        raise ValueError('Unknown supervised loss {}'.format(supervised_method))

########################################################################################################################

class SupervisedLoss(LossBase):
    """
    Supervised loss for inverse depth maps.

    Parameters
    ----------
    supervised_method : str
        Which supervised method will be used
    supervised_num_scales : int
        Number of scales used by the supervised loss
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_method='sparse-l1',
                 supervised_num_scales=4, progressive_scaling=0.0, **kwargs):
        super().__init__()
        self.loss_func = get_loss_func(supervised_method)
        self.supervised_method = supervised_method
        self.n = supervised_num_scales
        self.progressive_scaling = ProgressiveScaling(
            progressive_scaling, self.n)

    ########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'supervised_num_scales': self.n,
        }

########################################################################################################################

    def calculate_loss(self, inv_depths, gt_inv_depths):
        """
        Calculate the supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            List of predicted inverse depth maps
        gt_inv_depths : list of torch.Tensor [B,1,H,W]
            List of ground-truth inverse depth maps

        Returns
        -------
        loss : torch.Tensor [1]
            Average supervised loss for all scales
        """
        # If using a sparse loss, mask invalid pixels for all scales
        if self.supervised_method.startswith('sparse'):
            for i in range(self.n):
                mask = (gt_inv_depths[i] > 0.).detach()
                inv_depths[i] = inv_depths[i][mask]
                gt_inv_depths[i] = gt_inv_depths[i][mask]
        # Return per-scale average loss
        return sum([self.loss_func(inv_depths[i], gt_inv_depths[i])
                    for i in range(self.n)]) / self.n

    def forward(self, inv_depths, gt_inv_depth,
                return_logs=False, progress=0.0):
        """
        Calculates training supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        gt_inv_depth : torch.Tensor [B,1,H,W]
            Ground-truth depth map for the original image
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        # If using progressive scaling
        self.n = self.progressive_scaling(progress)
        # Match predicted scales for ground-truth
        gt_inv_depths = match_scales(gt_inv_depth, inv_depths, self.n,
                                     mode='nearest', align_corners=None)
        # Calculate and store supervised loss
        loss = self.calculate_loss(inv_depths, gt_inv_depths)
        self.add_metric('supervised_loss', loss)
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }
