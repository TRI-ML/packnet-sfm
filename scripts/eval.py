# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import torch

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.trainers.horovod_trainer import HorovodTrainer
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.horovod import hvd_init


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM evaluation script')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('--config', type=str, default=None, help='Configuration (.yaml)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.config is None or args.config.endswith('.yaml'), \
        'You need to provide a .yaml file as configuration'
    return args


def test(ckpt_file, cfg_file, half):
    """
    Monocular depth estimation test script.

    Parameters
    ----------
    ckpt_file : str
        Checkpoint path for a pretrained model
    cfg_file : str
        Configuration file
    half: bool
        use half precision (fp16)
    """
    # Initialize horovod
    hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(ckpt_file, cfg_file)

    # Set debug if requested
    set_debug(config.debug)

    # Initialize monodepth model from checkpoint arguments
    model_wrapper = ModelWrapper(config)
    # Restore model state
    model_wrapper.load_state_dict(state_dict)

    # change to half precision for evaluation if requested
    config.arch["dtype"] = torch.float16 if half else None

    # Create trainer with args.arch parameters
    trainer = HorovodTrainer(**config.arch)

    # Test model
    trainer.test(model_wrapper)


if __name__ == '__main__':
    args = parse_args()
    test(args.checkpoint, args.config, args.half)
