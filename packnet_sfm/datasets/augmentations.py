# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image

from packnet_sfm.utils.misc import filter_dict
from packnet_sfm.utils.types import is_seq

########################################################################################################################

def resize_image(image, shape, interpolation=Image.ANTIALIAS):
    """
    Resizes input image.

    Parameters
    ----------
    image : Image.PIL
        Input image
    shape : tuple [H,W]
        Output shape
    interpolation : int
        Interpolation mode

    Returns
    -------
    image : Image.PIL
        Resized image
    """
    transform = transforms.Resize(shape, interpolation=interpolation)
    return transform(image)

def resize_depth(depth, shape):
    """
    Resizes depth map.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W]
        Resized depth map
    """
    depth = cv2.resize(depth, dsize=shape[::-1],
                       interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(depth, axis=2)


def resize_depth_preserve(depth, shape):
    """
    Resizes depth map preserving all valid depth pixels
    Multiple downsampled points can be assigned to the same pixel.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W,1]
        Resized depth map
    """
    # Return if depth value is None
    if depth is None:
        return depth
    # If a single number is provided, use resize ratio
    if not is_seq(shape):
        shape = tuple(int(s * shape) for s in depth.shape)
    # Store dimensions and reshapes to single column
    depth = np.squeeze(depth)
    h, w = depth.shape
    x = depth.reshape(-1)
    # Create coordinate grid
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    # Filters valid points
    idx = x > 0
    crd, val = uv[idx], x[idx]
    # Downsamples coordinates
    crd[:, 0] = (crd[:, 0] * (shape[0] / h)).astype(np.int32)
    crd[:, 1] = (crd[:, 1] * (shape[1] / w)).astype(np.int32)
    # Filters points inside image
    idx = (crd[:, 0] < shape[0]) & (crd[:, 1] < shape[1])
    crd, val = crd[idx], val[idx]
    # Creates downsampled depth image and assigns points
    depth = np.zeros(shape)
    depth[crd[:, 0], crd[:, 1]] = val
    # Return resized depth map
    return np.expand_dims(depth, axis=2)


def resize_sample_image_and_intrinsics(sample, shape,
                                       image_interpolation=Image.ANTIALIAS):
    """
    Resizes the image and intrinsics of a sample

    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize image and corresponding intrinsics
    image_transform = transforms.Resize(shape, interpolation=image_interpolation)
    (orig_w, orig_h) = sample['rgb'].size
    (out_h, out_w) = shape
    # Scale intrinsics
    for key in filter_dict(sample, [
        'intrinsics'
    ]):
        intrinsics = np.copy(sample[key])
        intrinsics[0] *= out_w / orig_w
        intrinsics[1] *= out_h / orig_h
        sample[key] = intrinsics
    # Scale images
    for key in filter_dict(sample, [
        'rgb', 'rgb_original',
    ]):
        sample[key] = image_transform(sample[key])
    # Scale context images
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original',
    ]):
        sample[key] = [image_transform(k) for k in sample[key]]
    # Return resized sample
    return sample

def resize_sample(sample, shape, image_interpolation=Image.ANTIALIAS):
    """
    Resizes a sample, including image, intrinsics and depth maps.

    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize image and intrinsics
    sample = resize_sample_image_and_intrinsics(sample, shape, image_interpolation)
    # Resize depth maps
    for key in filter_dict(sample, [
        'depth', 'input_depth',
    ]):
        sample[key] = resize_depth_preserve(sample[key], shape)
    # Resize depth contexts
    for key in filter_dict(sample, [
        'depth_context',
    ]):
        sample[key] = [resize_depth_preserve(k, shape) for k in sample[key]]
    # Return resized sample
    return sample

########################################################################################################################

def to_tensor(image, tensor_type='torch.FloatTensor'):
    """Casts an image to a torch.Tensor"""
    transform = transforms.ToTensor()
    return transform(image).type(tensor_type)

def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.

    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to

    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """
    transform = transforms.ToTensor()
    # Convert single items
    for key in filter_dict(sample, [
        'rgb', 'rgb_original', 'depth', 'input_depth',
    ]):
        sample[key] = transform(sample[key]).type(tensor_type)
    # Convert lists
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original', 'depth_context'
    ]):
        sample[key] = [transform(k).type(tensor_type) for k in sample[key]]
    # Return converted sample
    return sample

########################################################################################################################

def duplicate_sample(sample):
    """
    Duplicates sample images and contexts to preserve their unaugmented versions.

    Parameters
    ----------
    sample : dict
        Input sample

    Returns
    -------
    sample : dict
        Sample including [+"_original"] keys with copies of images and contexts.
    """
    # Duplicate single items
    for key in filter_dict(sample, [
        'rgb'
    ]):
        sample['{}_original'.format(key)] = sample[key].copy()
    # Duplicate lists
    for key in filter_dict(sample, [
        'rgb_context'
    ]):
        sample['{}_original'.format(key)] = [k.copy() for k in sample[key]]
    # Return duplicated sample
    return sample

def colorjitter_sample(sample, parameters, prob=1.0):
    """
    Jitters input images as data augmentation.

    Parameters
    ----------
    sample : dict
        Input sample
    parameters : tuple (brightness, contrast, saturation, hue, color)
        Color jittering parameters
    prob : float
        Jittering probability

    Returns
    -------
    sample : dict
        Jittered sample
    """
    if random.random() < prob:
        # Prepare jitter transformation
        color_jitter_transform = random_color_jitter_transform(parameters[:4])
        # Prepare color transformation if requested
        if len(parameters) > 4 and parameters[4] > 0:
            matrix = (random.uniform(1. - parameters[4], 1 + parameters[4]), 0, 0, 0,
                      0, random.uniform(1. - parameters[4], 1 + parameters[4]), 0, 0,
                      0, 0, random.uniform(1. - parameters[4], 1 + parameters[4]), 0)
        else:
            matrix = None
        # Jitter single items
        for key in filter_dict(sample, [
            'rgb'
        ]):
            sample[key] = color_jitter_transform(sample[key])
            if matrix is not None:  # If applying color transformation
                sample[key] = sample[key].convert('RGB', matrix)
        # Jitter lists
        for key in filter_dict(sample, [
            'rgb_context'
        ]):
            sample[key] = [color_jitter_transform(k) for k in sample[key]]
            if matrix is not None:  # If applying color transformation
                sample[key] = [k.convert('RGB', matrix) for k in sample[key]]
    # Return jittered (?) sample
    return sample


def random_color_jitter_transform(parameters):
    """
    Creates a reusable color jitter transformation

    Parameters
    ----------
    parameters : tuple (brightness, contrast, saturation, hue, color)
        Color jittering parameters

    Returns
    -------
    transform : torch.vision.Transform
        Color jitter transformation with fixed parameters
    """
    # Get and unpack values
    brightness, contrast, saturation, hue = parameters
    brightness = [max(0, 1 - brightness), 1 + brightness]
    contrast = [max(0, 1 - contrast), 1 + contrast]
    saturation = [max(0, 1 - saturation), 1 + saturation]
    hue = [-hue, hue]

    # Initialize transformation list
    all_transforms = []

    # Add brightness transformation
    if brightness is not None:
        brightness_factor = random.uniform(brightness[0], brightness[1])
        all_transforms.append(transforms.Lambda(
            lambda img: transforms.functional.adjust_brightness(img, brightness_factor)))
    # Add contrast transformation
    if contrast is not None:
        contrast_factor = random.uniform(contrast[0], contrast[1])
        all_transforms.append(transforms.Lambda(
            lambda img: transforms.functional.adjust_contrast(img, contrast_factor)))
    # Add saturation transformation
    if saturation is not None:
        saturation_factor = random.uniform(saturation[0], saturation[1])
        all_transforms.append(transforms.Lambda(
            lambda img: transforms.functional.adjust_saturation(img, saturation_factor)))
    # Add hue transformation
    if hue is not None:
        hue_factor = random.uniform(hue[0], hue[1])
        all_transforms.append(transforms.Lambda(
            lambda img: transforms.functional.adjust_hue(img, hue_factor)))
    # Shuffle transformation order
    random.shuffle(all_transforms)
    # Return composed transformation
    return transforms.Compose(all_transforms)


def crop_image(image, borders):
    """
    Crop a PIL Image

    Parameters
    ----------
    image : PIL.Image
        Input image
    borders : tuple (left, top, right, bottom)
        Borders used for cropping

    Returns
    -------
    image : PIL.Image
        Cropped image
    """
    return image.crop(borders)


def crop_intrinsics(intrinsics, borders):
    """
    Crop camera intrinsics matrix

    Parameters
    ----------
    intrinsics : np.array [3,3]
        Original intrinsics matrix
    borders : tuple
        Borders used for cropping (left, top, right, bottom)
    Returns
    -------
    intrinsics : np.array [3,3]
        Cropped intrinsics matrix
    """
    intrinsics = np.copy(intrinsics)
    intrinsics[0, 2] -= borders[0]
    intrinsics[1, 2] -= borders[1]
    return intrinsics


def crop_depth(depth, borders):
    """
    Crop a numpy depth map

    Parameters
    ----------
    depth : np.array
        Input numpy array
    borders : tuple
        Borders used for cropping (left, top, right, bottom)

    Returns
    -------
    image : np.array
        Cropped numpy array
    """
    # Return if depth value is None
    if depth is None:
        return depth
    return depth[borders[1]:borders[3], borders[0]:borders[2]]


def crop_sample_input(sample, borders):
    """
    Crops the input information of a sample (i.e. that go to the networks)

    Parameters
    ----------
    sample : dict
        Dictionary with sample values (output from a dataset's __getitem__ method)
    borders : tuple
        Borders used for cropping (left, top, right, bottom)

    Returns
    -------
    sample : dict
        Cropped sample
    """
    # Crop intrinsics
    for key in filter_dict(sample, [
        'intrinsics'
    ]):
        # Create copy of full intrinsics
        if key + '_full' not in sample.keys():
            sample[key + '_full'] = np.copy(sample[key])
        sample[key] = crop_intrinsics(sample[key], borders)
    # Crop images
    for key in filter_dict(sample, [
        'rgb', 'rgb_original', 'warped_rgb',
    ]):
        sample[key] = crop_image(sample[key], borders)
    # Crop context images
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original',
    ]):
        sample[key] = [crop_image(val, borders) for val in sample[key]]
    # Crop input depth maps
    for key in filter_dict(sample, [
        'input_depth', 'bbox2d_depth', 'bbox3d_depth'
    ]):
        sample[key] = crop_depth(sample[key], borders)
    # Crop context input depth maps
    for key in filter_dict(sample, [
        'input_depth_context',
    ]):
        sample[key] = [crop_depth(val, borders) for val in sample[key]]
    # Return cropped sample
    return sample


def crop_sample_supervision(sample, borders):
    """
    Crops the output information of a sample (i.e. ground-truth supervision)

    Parameters
    ----------
    sample : dict
        Dictionary with sample values (output from a dataset's __getitem__ method)
    borders : tuple
        Borders used for cropping

    Returns
    -------
    sample : dict
        Cropped sample
    """
    # Crop maps
    for key in filter_dict(sample, [
        'depth', 'bbox2d_depth', 'bbox3d_depth', 'semantic',
        'bwd_optical_flow', 'fwd_optical_flow', 'valid_fwd_optical_flow',
        'bwd_scene_flow', 'fwd_scene_flow',
    ]):
        sample[key] = crop_depth(sample[key], borders)
    # Crop context maps
    for key in filter_dict(sample, [
        'depth_context', 'semantic_context',
        'bwd_optical_flow_context', 'fwd_optical_flow_context',
        'bwd_scene_flow_context', 'fwd_scene_flow_context',
    ]):
        sample[key] = [crop_depth(k, borders) for k in sample[key]]
    # Return cropped sample
    return sample


def crop_sample(sample, borders):
    """
    Crops a sample, including image, intrinsics and depth maps.

    Parameters
    ----------
    sample : dict
        Dictionary with sample values (output from a dataset's __getitem__ method)
    borders : tuple
        Borders used for cropping

    Returns
    -------
    sample : dict
        Cropped sample
    """
    # Crop input information
    sample = crop_sample_input(sample, borders)
    # Crop output information
    sample = crop_sample_supervision(sample, borders)
    # Return cropped sample
    return sample