
import argparse
import numpy as np
import os
import torch

from glob import glob
from argparse import Namespace
from packnet_sfm.utils.depth import load_depth
from tqdm import tqdm

from packnet_sfm.utils.depth import load_depth, compute_depth_metrics


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


def main(args):
    # Get and sort ground-truth and predicted files
    exts = ('npz', 'png')
    gt_files, pred_files = [], []
    for ext in exts:
        gt_files.extend(glob(os.path.join(args.gt_folder, '*.{}'.format(ext))))
        pred_files.extend(glob(os.path.join(args.pred_folder, '*.{}'.format(ext))))
    # Sort ground-truth and prediction
    gt_files.sort()
    pred_files.sort()
    # Loop over all files
    metrics = []
    progress_bar = tqdm(zip(gt_files, pred_files), total=len(gt_files))
    for gt, pred in progress_bar:
        # Get and prepare ground-truth and predictions
        gt = torch.tensor(load_depth(gt)).unsqueeze(0).unsqueeze(0)
        pred = torch.tensor(load_depth(pred)).unsqueeze(0).unsqueeze(0)
        # Calculate metrics
        metrics.append(compute_depth_metrics(
            args, gt, pred, use_gt_scale=args.use_gt_scale))
    # Get and print average value
    metrics = (sum(metrics) / len(metrics)).detach().cpu().numpy()
    names = ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    for name, metric in zip(names, metrics):
        print('{} = {}'.format(name, metric))


if __name__ == '__main__':
    args = parse_args()
    main(args)
