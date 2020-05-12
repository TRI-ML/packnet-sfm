# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
import argparse
from glob import glob
from cv2 import imwrite
import numpy as np

from packnet_sfm import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import viz_inv_depth
from packnet_sfm.utils.logging import pcolor


def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM evaluation script')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('--input', type=str, help='Input file or folder')
    parser.add_argument('--output', type=str, help='Output file or foler')
    parser.add_argument('--image_shape', type=tuple, default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (is_image(args.input) and is_image(args.output)) or \
           (not is_image(args.input) and not is_image(args.input)), \
        'Input and output must both be images or folders'
    return args


def process(input_file, output_file, model_wrapper, image_shape):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file : str
        Image file
    output_file : str
        Output file, or folder where the output will be saved
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape

    Returns
    -------

    """
    # Load image
    image = load_image(input_file)
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()))

    # Depth inference
    depth = model_wrapper.depth(image)[0]

    # Prepare RGB image
    rgb_i = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    # Prepare inverse depth
    pred_inv_depth_i = viz_inv_depth(depth[0]) * 255
    # Concatenate both vertically
    image = np.concatenate([rgb_i, pred_inv_depth_i], 0)
    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))
    # Save visualization
    print('Saving {} to {}'.format(
        pcolor(input_file, 'cyan', attrs=['bold']),
        pcolor(output_file, 'magenta', attrs=['bold'])))
    imwrite(output_file, image[:, :, ::-1])


def infer(ckpt_file, input_file, output_file, image_shape):
    """
    Monocular depth estimation test script.

    Parameters
    ----------
    ckpt_file : str
        Checkpoint path for a pretrained model
    input_file : str
        File or folder with input images
    output_file : str
        File or folder with output images
    image_shape : tuple
        Input image shape (H,W)
    """
    # Initialize horovod
    hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(ckpt_file)

    # If no image shape is provided, use the checkpoint one
    if image_shape is None:
        image_shape = config.datasets.augmentation.image_shape

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()))

    if os.path.isdir(input_file):
        # If input file is a folder, search for image files
        files = []
        for ext in ['png', 'jpg']:
            files.extend(glob((os.path.join(input_file, '*.{}'.format(ext)))))
        files.sort()
        print0('Found {} files'.format(len(files)))
    else:
        # Otherwise, use it as is
        files = [input_file]

    # Process each file
    for file in files[rank()::world_size()]:
        process(file, output_file, model_wrapper, image_shape)


if __name__ == '__main__':
    args = parse_args()
    infer(args.checkpoint, args.input, args.output, args.image_shape)
