# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import partial
from packnet_sfm.datasets.augmentations import resize_image, resize_sample, resize_depth, \
    duplicate_sample, colorjitter_sample, to_tensor_sample, crop_sample, crop_sample_input, resize_depth_preserve
from packnet_sfm.utils.misc import parse_crop_borders

########################################################################################################################

def train_transforms(sample, image_shape, jittering, crop_train_borders):
    """
    Training data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    crop_train_borders : tuple (left, top, right, down)
        Border for cropping

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(crop_train_borders) > 0:
        borders = parse_crop_borders(crop_train_borders, sample['rgb'].size[::-1])
        sample = crop_sample(sample, borders)
    if len(image_shape) > 0:
        sample = resize_sample(sample, image_shape)
    sample = duplicate_sample(sample)
    if len(jittering) > 0:
        sample = colorjitter_sample(sample, jittering)
    sample = to_tensor_sample(sample)
    return sample

def validation_transforms(sample, image_shape, crop_eval_borders):
    """
    Validation data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    crop_eval_borders : tuple (left, top, right, down)
        Border for cropping

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(crop_eval_borders) > 0:
        borders = parse_crop_borders(crop_eval_borders, sample['rgb'].size[::-1])
        sample = crop_sample_input(sample, borders)
    if len(image_shape) > 0:
        sample['rgb'] = resize_image(sample['rgb'], image_shape)
        if 'input_depth' in sample:
            sample['input_depth'] = resize_depth_preserve(sample['input_depth'], image_shape)
    sample = to_tensor_sample(sample)
    return sample

def test_transforms(sample, image_shape, crop_eval_borders):
    """
    Test data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(crop_eval_borders) > 0:
        borders = parse_crop_borders(crop_eval_borders, sample['rgb'].size[::-1])
        sample = crop_sample_input(sample, borders)
    if len(image_shape) > 0:
        sample['rgb'] = resize_image(sample['rgb'], image_shape)
        if 'input_depth' in sample:
            sample['input_depth'] = resize_depth(sample['input_depth'], image_shape)
    sample = to_tensor_sample(sample)
    return sample

def get_transforms(mode, image_shape, jittering, crop_train_borders,
                   crop_eval_borders, **kwargs):
    """
    Get data augmentation transformations for each split

    Parameters
    ----------
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the data augmentation transformations
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    crop_train_borders : tuple (left, top, right, down)
        Border for cropping
    crop_eval_borders : tuple (left, top, right, down)
        Border for cropping

    Returns
    -------
        XXX_transform: Partial function
            Data augmentation transformation for that mode
    """
    if mode == 'train':
        return partial(train_transforms,
                       image_shape=image_shape,
                       jittering=jittering,
                       crop_train_borders=crop_train_borders)
    elif mode == 'validation':
        return partial(validation_transforms,
                       crop_eval_borders=crop_eval_borders,
                       image_shape=image_shape)
    elif mode == 'test':
        return partial(test_transforms,
                       crop_eval_borders=crop_eval_borders,
                       image_shape=image_shape)
    else:
        raise ValueError('Unknown mode {}'.format(mode))

########################################################################################################################

