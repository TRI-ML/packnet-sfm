# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
Saving utilities
"""

import numpy as np
import os
import cv2
from monodepth.vis import vis_inverse_depth


class Saver:
    """Saves inference data for later use"""
    def __init__(self, args):
        """
        Initializes the saver. The args parameter needs to have:
        - pretrained_model: pretrained_model (uses to extract folder name for saving)
        - input_path: input directory/split (uses to extract folder name for saving)
        - save_output: root path where data will be saved
        """
        assert 'input_path' and 'pretrained_model' in args
        model_name = args.pretrained_model.split('/')[-1].split('.')[0]
        split_name = args.input_path.split('/')[-1].split('.')[0]
        self.path = os.path.join(args.save_output, split_name, model_name)
        os.makedirs(self.path, exist_ok=True)

    def save(self, sample, pred_inv_depth, i):
        """
        Saves data from one step. This includes:
        - image with RGB and Inverse Depth visualization
        - NPZ file with predicted depths
        """
        rgb = sample['left_image'].cuda()
        rgb = rgb[0].detach().cpu().numpy().transpose(1, 2, 0)

        pred_depth = 1. / pred_inv_depth
        pred_depth = pred_depth[0].detach().cpu().numpy()
        pred_depth_vis = vis_inverse_depth(1. / pred_depth[0])[:, :, ::-1] * 255

        rgb, vis = rgb[:, :, ::-1] * 255, pred_depth_vis
        img = np.concatenate([rgb, vis], 0)

        cv2.imwrite(os.path.join(self.path, '%09d.jpg' % i), img)
        np.savez_compressed(os.path.join(self.path, '%09d.npz' % i), pred_depth)

