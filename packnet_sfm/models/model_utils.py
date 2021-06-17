# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.utils.image import flip_lr, interpolate_scales
from packnet_sfm.utils.misc import filter_dict
from packnet_sfm.utils.types import is_tensor, is_list, is_numpy


def flip(tensor, flip_fn):
    """
    Flip tensors or list of tensors based on a function

    Parameters
    ----------
    tensor : torch.Tensor or list[torch.Tensor] or list[list[torch.Tensor]]
        Tensor to be flipped
    flip_fn : Function
        Flip function

    Returns
    -------
    tensor : torch.Tensor or list[torch.Tensor] or list[list[torch.Tensor]]
        Flipped tensor or list of tensors
    """
    if not is_list(tensor):
        return flip_fn(tensor)
    else:
        if not is_list(tensor[0]):
            return [flip_fn(val) for val in tensor]
        else:
            return [[flip_fn(v) for v in val] for val in tensor]


def merge_outputs(*outputs):
    """
    Merges model outputs for logging

    Parameters
    ----------
    outputs : tuple of dict
        Outputs to be merged

    Returns
    -------
    output : dict
        Dictionary with a "metrics" key containing a dictionary with various metrics and
        all other keys that are not "loss" (it is handled differently).
    """
    ignore = ['loss'] # Keys to ignore
    combine = ['metrics'] # Keys to combine
    merge = {key: {} for key in combine}
    for output in outputs:
        # Iterate over all keys
        for key, val in output.items():
            # Combine these keys
            if key in combine:
                for sub_key, sub_val in output[key].items():
                    assert sub_key not in merge[key].keys(), \
                        'Combining duplicated key {} to {}'.format(sub_key, key)
                    merge[key][sub_key] = sub_val
            # Ignore these keys
            elif key not in ignore:
                assert key not in merge.keys(), \
                    'Adding duplicated key {}'.format(key)
                merge[key] = val
    return merge


def stack_batch(batch):
    """
    Stack multi-camera batches (B,N,C,H,W becomes BN,C,H,W)

    Parameters
    ----------
    batch : dict
        Batch

    Returns
    -------
    batch : dict
        Stacked batch
    """
    # If there is multi-camera information
    if len(batch['rgb'].shape) == 5:
        assert batch['rgb'].shape[0] == 1, 'Only batch size 1 is supported for multi-cameras'
        # Loop over all keys
        for key in batch.keys():
            # If list, stack every item
            if is_list(batch[key]):
                if is_tensor(batch[key][0]) or is_numpy(batch[key][0]):
                    batch[key] = [sample[0] for sample in batch[key]]
            # Else, stack single item
            else:
                batch[key] = batch[key][0]
    return batch


def flip_batch_input(batch):
    """
    Flip batch input information (copies data first)

    Parameters
    ----------
    batch : dict
        Batch information

    Returns
    -------
    batch : dict
        Flipped batch
    """
    # Flip tensors
    for key in filter_dict(batch, [
        'rgb', 'rgb_context',
        'input_depth', 'input_depth_context',
    ]):
        batch[key] = flip(batch[key], flip_lr)
    # Flip intrinsics
    for key in filter_dict(batch, [
        'intrinsics'
    ]):
        batch[key] = batch[key].clone()
        batch[key][:, 0, 2] = batch['rgb'].shape[3] - batch[key][:, 0, 2]
    # Return flipped batch
    return batch


def flip_output(output):
    """
    Flip output information

    Parameters
    ----------
    output : dict
        Dictionary of model outputs (e.g. with keys like 'inv_depths' and 'uncertainty')

    Returns
    -------
    output : dict
        Flipped output
    """
    # Flip tensors
    for key in filter_dict(output, [
        'uncertainty', 'logits_semantic', 'ord_probability',
        'inv_depths', 'inv_depths_context', 'inv_depths1', 'inv_depths2',
        'pred_depth', 'pred_depth_context', 'pred_depth1', 'pred_depth2',
        'pred_inv_depth', 'pred_inv_depth_context', 'pred_inv_depth1', 'pred_inv_depth2',
    ]):
        output[key] = flip(output[key], flip_lr)
    return output


def upsample_output(output, mode='nearest', align_corners=None):
    """
    Upsample multi-scale outputs to full resolution.

    Parameters
    ----------
    output : dict
        Dictionary of model outputs (e.g. with keys like 'inv_depths' and 'uncertainty')
    mode : str
        Which interpolation mode is used
    align_corners: bool or None
        Whether corners will be aligned during interpolation

    Returns
    -------
    output : dict
        Upsampled output
    """
    for key in filter_dict(output, [
        'inv_depths', 'uncertainty'
    ]):
        output[key] = interpolate_scales(
            output[key], mode=mode, align_corners=align_corners)
    for key in filter_dict(output, [
        'inv_depths_context'
    ]):
        output[key] = [interpolate_scales(
            val, mode=mode, align_corners=align_corners) for val in output[key]]
    return output