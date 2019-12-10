# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
Data augmentation functions
"""

import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def filter_dict(dict, keywords):
    """
    Returns only keywords that are present in a dictionary
    """
    return [key for key in keywords if key in dict]


def resize_sample_image_and_intrinsics(sample, image_shape, image_interpolation=Image.ANTIALIAS):
    """
    Takes a sample and resizes the input image ['left_image'].
    It also resizes the corresponding camera intrinsics ['left_intrinsics'] and ['right_intrinsics']
    """
    # Resize image and corresponding intrinsics
    image_transform = transforms.Resize(image_shape, interpolation=image_interpolation)
    original_shape = sample['left_image'].size
    (orig_w, orig_h) = original_shape
    (out_h, out_w) = image_shape

    for key in filter_dict(sample, [
        'left_intrinsics', 'right_intrinsics'
    ]):
        # Note this is swapped here because PIL.Image.size -> (w,h)
        # but we specify image_shape -> (h,w) for rescaling
        y_scale = out_h / orig_h
        x_scale = out_w / orig_w
        # scale fx and fy appropriately
        intrinsics = np.copy(sample[key])
        intrinsics[0] *= x_scale
        intrinsics[1] *= y_scale
        sample[key] = intrinsics

    # Scale image (default is antialias)
    for key in filter_dict(sample, [
        'left_image', 'right_image',
    ]):
        sample[key] = image_transform(sample[key])

    return sample


def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Converts all fields from a sample to tensor.
    """
    transform = transforms.ToTensor()
    for key in filter_dict(sample, [
            'left_image', 'right_image',
            'left_depth', 'right_depth',
    ]):
        sample[key] = transform(sample[key]).type(tensor_type)
    return sample
