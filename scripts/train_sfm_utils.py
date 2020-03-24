# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
from monodepth.models import monodepth_beta, load_net_from_checkpoint
from monodepth.functional.image import scale_image
import os


def load_dispnet_with_args(args):
    """
    Loads a pretrained depth network
    """
    checkpoint = torch.load(args.pretrained_model)
    # check for relevant args
    assert 'args' in checkpoint, 'Cannot find args in checkpoint.'
    checkpoint_args = checkpoint['args']
    for arg in ['disp_model', 'dropout', 'input_height', 'input_width']:
        assert arg in checkpoint_args, 'Could not find argument {}'.format(arg)
    disp_net = monodepth_beta(checkpoint_args.disp_model,
                              dropout=checkpoint_args.dropout)
    disp_net = load_net_from_checkpoint(disp_net, args.pretrained_model, starts_with='disp_network')
    disp_net = disp_net.cuda()  # move to GPU
    print('Loaded disp net of type {}'.format(checkpoint_args.disp_model))

    return disp_net, checkpoint_args


def compute_depth_errors(args, gt, pred, use_gt_scale=True, crop=True):
    """
    Computes depth errors given ground-truth and predicted depths
    use_gt_scale: If True, median ground-truth scaling is used
    crop: If True, apply a crop in the image before evaluating
    """
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    rmse, rmse_log = 0, 0

    batch_size, _, gt_height, gt_width = gt.shape
    pred = scale_image(pred, gt_height, gt_width, mode='bilinear', align_corners=True)
    for current_gt, current_pred in zip(gt, pred):
        gt_channels, gt_height, gt_width = current_gt.shape
        current_gt = torch.squeeze(current_gt)
        current_pred = torch.squeeze(current_pred)

        # Mask within min and max depth
        valid = (current_gt > args.min_depth) & (current_gt < args.max_depth)

        if crop:
            # crop used by Garg ECCV16 to reproduce Eigen NIPS14 results
            # construct a mask of False values, with the same size as target
            # and then set to True values inside the crop
            crop_mask = torch.zeros(current_gt.shape).byte().cuda()
            y1, y2 = int(0.40810811 * gt_height), int(0.99189189 * gt_height)
            x1, x2 = int(0.03594771 * gt_width), int(0.96405229 * gt_width)
            crop_mask[y1:y2, x1:x2] = 1
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid]

        if use_gt_scale:
            # Median ground-truth scaling
            valid_pred = valid_pred * torch.median(valid_gt) / torch.median(valid_pred)

        valid_pred = valid_pred.clamp(args.min_depth, args.max_depth)

        # Calculates threshold values
        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25**2).float().mean()
        a3 += (thresh < 1.25**3).float().mean()

        # Calculates absolute relative error
        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        # Calculates square relative error
        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

        # Calculates root mean square error and its log
        rmse += torch.sqrt(torch.mean((valid_gt - valid_pred)**2))
        r_log = (torch.log(valid_gt) - torch.log(valid_pred))**2
        rmse_log += torch.sqrt(torch.mean(r_log))

    return torch.tensor([metric / batch_size for metric in [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]])
