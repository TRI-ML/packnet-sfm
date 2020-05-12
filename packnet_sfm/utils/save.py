# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import os

from packnet_sfm.utils.logging import prepare_dataset_prefix
from packnet_sfm.utils.depth import inv2depth, viz_inv_depth

########################################################################################################################

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

    # If we want to save depth maps
    if save.viz or save.npz:
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
            if save.npz:
                # Get depth from predicted depth map and save to .npz
                np.savez_compressed('{}/{}.npz'.format(save_path, filename[i]),
                    depth=inv2depth(pred_inv_depth[i]).squeeze().detach().cpu().numpy())
            # Save inverse depth visualizations
            if save.viz:
                # Prepare RGB image
                rgb_i = rgb[i].permute(1, 2, 0).detach().cpu().numpy() * 255
                # Prepare inverse depth
                pred_inv_depth_i = viz_inv_depth(pred_inv_depth[i]) * 255
                # Concatenate both vertically
                image = np.concatenate([rgb_i, pred_inv_depth_i], 0)
                # Write to disk
                cv2.imwrite('{}/{}.png'.format(
                    save_path, filename[i]), image[:, :, ::-1])

########################################################################################################################