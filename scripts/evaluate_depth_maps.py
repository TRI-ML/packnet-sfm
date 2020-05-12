
import os
from glob import glob
import numpy as np
import torch
from argparse import Namespace

from packnet_sfm.utils.depth import compute_depth_metrics
import argparse


def parse_args():
    """Parse arguments for benchmark script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM benchmark script')
    parser.add_argument('--pred_folder', type=str,
                        help='Folder containing predicted depth maps (.npz with key "depth")')
    parser.add_argument('--gt_folder', type=str,
                        help='Folder containing ground-truth depth maps (.npz with key "depth")')
    parser.add_argument('--use_gt_scale', action='store_true',
                        help='Use ground-truth median scaling on predicted depth maps')
    parser.add_argument('--min_depth', type=float, default=0.,
                        help='Minimum distance to consider during evaluation')
    parser.add_argument('--max_depth', type=float, default=80.,
                        help='Maximum distance to consider during evaluation')
    parser.add_argument('--crop', type=str, default='', choices=['', 'garg'],
                        help='Which crop to use during evaluation')
    args = parser.parse_args()
    return args


def evaluate_depth_maps(pred_folder, gt_folder, use_gt_scale, **kwargs):
    """
    Calculates depth metrics from a folder of predicted and ground-truth depth files

    Parameters
    ----------
    pred_folder : str
        Folder containing predicted depth maps (.npz with key 'depth')
    gt_folder : str
        Folder containing ground-truth depth maps (.npz with key 'depth')
    use_gt_scale : bool
        Using ground-truth median scaling or not
    kwargs : dict
        Extra parameters for depth evaluation
    """
    # Get and sort ground-truth files
    gt_files = glob(os.path.join(gt_folder, '*.npz'))
    gt_files.sort()
    # Get and sort predicted files
    pred_files = glob(os.path.join(pred_folder, '*.npz'))
    pred_files.sort()
    # Prepare configuration
    config = Namespace(**kwargs)
    # Loop over all files
    metrics = []
    for gt, pred in zip(gt_files, pred_files):
        # Get and prepare ground-truth
        gt = np.load(gt)['depth']
        gt = torch.tensor(gt).unsqueeze(0).unsqueeze(0)
        # Get and prepare predictions
        pred = np.load(pred)['depth']
        pred = torch.tensor(pred).unsqueeze(0).unsqueeze(0)
        # Calculate metrics
        metrics.append(compute_depth_metrics(config, gt, pred,
                                             use_gt_scale=use_gt_scale))
    # Get and print average value
    metrics = (sum(metrics) / len(metrics)).detach().cpu().numpy()
    names = ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    for name, metric in zip(names, metrics):
        print('{} = {}'.format(name, metric))

if __name__ == '__main__':
    args = parse_args()
    evaluate_depth_maps(args.pred_folder, args.gt_folder,
                        use_gt_scale=args.use_gt_scale,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                        crop=args.crop)


