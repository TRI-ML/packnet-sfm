# Copyright 2020 Toyota Research Institute.  All rights reserved.

import numpy as np
import os

from packnet_sfm.utils.image import write_image
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import prepare_dataset_prefix


def save_depth(batch, output, args, dataset, save):
    """
    Save depth predictions in various ways

    Parameters
    ----------
    batch : dict
        Batch from dataloader
    output : dict
        Output from model
    args : tuple
        Step arguments
    dataset : CfgNode
        Dataset configuration
    save : CfgNode
        Save configuration
    """
    # If there is no save folder, don't save
    if save.folder is '':
        return

    # If we want to save
    if save.depth.rgb or save.depth.viz or save.depth.npz or save.depth.png:
        # Retrieve useful tensors
        rgb = batch['rgb']
        pred_inv_depth = output['inv_depth']

        # Prepare path strings
        filename = batch['filename']
        dataset_idx = 0 if len(args) == 1 else args[1]
        save_path = os.path.join(save.folder, 'depth',
                                 prepare_dataset_prefix(dataset, dataset_idx),
                                 os.path.basename(save.pretrained).split('.')[0])
        # Create folder
        os.makedirs(save_path, exist_ok=True)

        # For each image in the batch
        length = rgb.shape[0]
        for i in range(length):
            # Save numpy depth maps
            if save.depth.npz:
                write_depth('{}/{}_depth.npz'.format(save_path, filename[i]),
                            depth=inv2depth(pred_inv_depth[i]),
                            intrinsics=batch['intrinsics'][i] if 'intrinsics' in batch else None)
            # Save png depth maps
            if save.depth.png:
                write_depth('{}/{}_depth.png'.format(save_path, filename[i]),
                            depth=inv2depth(pred_inv_depth[i]))
            # Save rgb images
            if save.depth.rgb:
                rgb_i = rgb[i].permute(1, 2, 0).detach().cpu().numpy() * 255
                write_image('{}/{}_rgb.png'.format(save_path, filename[i]), rgb_i)
            # Save inverse depth visualizations
            if save.depth.viz:
                viz_i = viz_inv_depth(pred_inv_depth[i]) * 255
                write_image('{}/{}_viz.png'.format(save_path, filename[i]), viz_i)
