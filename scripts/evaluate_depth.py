# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
Evaluation script for depth estimation from monocular images
Takes a pretrained pytorch model and calculates error metrics
==========================================================
Usage:
make docker-evaluate-depth \
MODEL=path/to/model.pth.tar (pretrained depth network) \
INPUT_PATH=/path/to/split (for now only a KITTI split file or an image folder are supported) \
DEPTH_TYPE=depth type used for evaluation [velodyne] \
CROP=crop used in depth evaluation [garg] \
SAVE_OUTPUT=/path/to/output (path where the output images and predicted depths will be saved)
==========================================================
"""
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.train_sfm_utils import compute_depth_errors, load_dispnet_with_args

from monodepth.datasets.data_augmentation import resize_sample_image_and_intrinsics, to_tensor_sample
from monodepth.functional.image import fliplr, get_resized_depth
from monodepth.logging import print_error_metrics
from monodepth.utils import post_process_disparity_with_border_ramps
from monodepth.save import Saver


def main():
    parser = argparse.ArgumentParser(description='TRI Monocular Depth Inference Script')
    parser.add_argument("--pretrained_model", type=str, help="pretrained model path")
    parser.add_argument("--input_path", required=True, type=str, help="evaluation split")
    parser.add_argument("--depth_type", type=str, default=None, help="depth type for evaluation")
    parser.add_argument("--min_depth", type=float, default=1e-3, help="min depth for evaluation")
    parser.add_argument("--max_depth", type=float, default=80., help="max depth for evaluation")
    parser.add_argument("--crop", type=str, default=None, help="crop used for evaluation")
    parser.add_argument("--save_output", type=str, default=None, help="Save evaluation data")
    args = parser.parse_args()

    # Fix None arguments
    args.crop = None if args.crop == 'None' else args.crop
    args.depth_type = None if args.depth_type == 'None' else args.depth_type
    args.save_output = None if args.save_output == 'None' else args.save_output

    # Loads disp_net model and checkpoint_args
    disp_net, checkpoint_args = load_dispnet_with_args(args)

    def test_transforms(sample):
        """
        Sample transformations.
        Includes resizing image/intrinsics and conversion to tensor.
        """
        sample = resize_sample_image_and_intrinsics(
            sample, image_shape=(checkpoint_args.input_height, checkpoint_args.input_width))
        sample = to_tensor_sample(sample)
        return sample

    # If a split is provided (ending with .txt)
    if args.input_path.endswith('.txt'):
        # Use KITTI loader
        checkpoint_args.depth_type = args.depth_type
        root_path, file_list = '/'.join(args.input_path.split('/')[:-2]), args.input_path
        from monodepth.datasets.kitti_context_loader import KittiContextLoader
        dataset = KittiContextLoader(root_path, file_list, 
                                     data_transform=test_transforms,
                                     depth_type=args.depth_type)
    else:
        # Otherwise, we assume it's a folder and use Image Sequence Loader
        from monodepth.datasets.image_sequence import ImageSequenceLoader
        dataset = ImageSequenceLoader(args.input_path,
                                      data_transform=test_transforms)

    # Prepare data loader
    data_loader = DataLoader(dataset, batch_size=1, pin_memory=False, shuffle=False,
                             num_workers=8, worker_init_fn=None, sampler=None)
    print('Loaded {} images '.format(len(data_loader)))

    # Depth metrics information
    error_names = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    error_descriptions = ['no_pp', 'with_pp', 'no_pp_gt', 'with_pp_gt']
    error_colors = ['magenta', 'cyan', 'magenta', 'cyan']
    resize_mode = 'bilinear'

    # Initialize error matrix
    n_errors = len(error_names)
    n_samples = len(data_loader.dataset)
    n_descriptions = len(error_descriptions)
    errors = torch.zeros(n_descriptions, n_samples, n_errors).cuda()

    # Apply crop if required
    if args.crop is not None:
        if args.crop == 'garg':
            args.crop = [0.03594771,  0.96405229, 0.40810811, 0.99189189]
        else:
            raise NotImplementedError('The crop {} is not available'.format(args.crop))

    # Prepare savers
    saver = None if args.save_output is None else Saver(args)

    # Iterate and compute depth
    disp_net.eval()
    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="infer_depth"):
            # Get regular and flipped images
            img = sample['left_image'].cuda()
            img_flipped = fliplr(img)
            # compute regular && flipped prediction
            inv_depth = disp_net(img)
            inv_depth_flipped = disp_net(img_flipped)

            # Calculated post-processed inverse depth
            inv_depth_postprocess = post_process_disparity_with_border_ramps(
                inv_depth, inv_depth_flipped, method='mean')

            # If saving, save it
            if args.save_output is not None:
                saver.save(sample, inv_depth_postprocess, i)

            # If we have depth, use it to calculate the errors
            if 'left_depth' in sample:
                gt_depth = sample['left_depth'].cuda()

                # Resize predicted depths to ground-truth resolution
                pred_depth = get_resized_depth(
                    inv_depth, gt_depth.shape, mode=resize_mode)
                pred_depth_postprocess = get_resized_depth(
                    inv_depth_postprocess, gt_depth.shape, mode=resize_mode)

                # With and without post-processing, with and without median ground-truth scaling
                errors[0, i] = compute_depth_errors(args, gt_depth, pred_depth,
                                                    use_gt_scale=False, crop=args.crop).cuda()
                errors[1, i] = compute_depth_errors(args, gt_depth, pred_depth_postprocess,
                                                    use_gt_scale=False, crop=args.crop).cuda()
                errors[2, i] = compute_depth_errors(args, gt_depth, pred_depth,
                                                    use_gt_scale=True, crop=args.crop).cuda()
                errors[3, i] = compute_depth_errors(args, gt_depth, pred_depth_postprocess,
                                                    use_gt_scale=True, crop=args.crop).cuda()

    # Print final depth metrics
    for i in range(n_descriptions):
        print_error_metrics(error_names, errors[i].mean(0), error_colors[i], error_descriptions[i])


if __name__ == '__main__':
    main()