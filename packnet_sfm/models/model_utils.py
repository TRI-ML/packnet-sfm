# Copyright 2020 Toyota Research Institute.  All rights reserved.


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
