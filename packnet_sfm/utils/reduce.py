
import torch
import numpy as np
from collections import OrderedDict
from packnet_sfm.utils.horovod import reduce_value
from packnet_sfm.utils.logging import prepare_dataset_prefix


def reduce_dict(data, to_item=False):
    """
    Reduce the mean values of a dictionary from all GPUs

    Parameters
    ----------
    data : dict
        Dictionary to be reduced
    to_item : bool
        True if the reduced values will be return as .item()

    Returns
    -------
    dict : dict
        Reduced dictionary
    """
    for key, val in data.items():
        data[key] = reduce_value(data[key], average=True, name=key)
        if to_item:
            data[key] = data[key].item()
    return data

def all_reduce_metrics(output_data_batch, datasets, name='depth'):
    """
    Reduce metrics for all batches and all datasets using Horovod

    Parameters
    ----------
    output_data_batch : list
        List of outputs for each batch
    datasets : list
        List of all considered datasets
    name : str
        Name of the task for the metric

    Returns
    -------
    all_metrics_dict : list
        List of reduced metrics
    """
    # If there is only one dataset, wrap in a list
    if isinstance(output_data_batch[0], dict):
        output_data_batch = [output_data_batch]
    # Get metrics keys and dimensions
    names = [key for key in list(output_data_batch[0][0].keys()) if key.startswith(name)]
    dims = [output_data_batch[0][0][name].shape[0] for name in names]
    # List storing metrics for all datasets
    all_metrics_dict = []
    # Loop over all datasets and all batches
    for output_batch, dataset in zip(output_data_batch, datasets):
        metrics_dict = OrderedDict()
        length = len(dataset)
        # Count how many times each sample was seen
        seen = torch.zeros(length)
        for output in output_batch:
            for i, idx in enumerate(output['idx']):
                seen[idx] += 1
        seen = reduce_value(seen, average=False, name='idx')
        assert not np.any(seen.numpy() == 0), \
            'Not all samples were seen during evaluation'
        # Reduce all relevant metrics
        for name, dim in zip(names, dims):
            metrics = torch.zeros(length, dim)
            for output in output_batch:
                for i, idx in enumerate(output['idx']):
                    metrics[idx] = output[name]
            metrics = reduce_value(metrics, average=False, name=name)
            metrics_dict[name] = (metrics / seen.view(-1, 1)).mean(0)
        # Append metrics dictionary to the list
        all_metrics_dict.append(metrics_dict)
    # Return list of metrics dictionary
    return all_metrics_dict

########################################################################################################################

def collate_metrics(output_data_batch, name='depth'):
    """
    Collate epoch output to produce average metrics

    Parameters
    ----------
    output_data_batch : list
        List of outputs for each batch
    name : str
        Name of the task for the metric

    Returns
    -------
    metrics_data : list
        List of collated metrics
    """
    # If there is only one dataset, wrap in a list
    if isinstance(output_data_batch[0], dict):
        output_data_batch = [output_data_batch]
    # Calculate the mean of all metrics
    metrics_data = []
    # For all datasets
    for i, output_batch in enumerate(output_data_batch):
        metrics = OrderedDict()
        # For all keys (assume they are the same for all batches)
        for key, val in output_batch[0].items():
            if key.startswith(name):
                metrics[key] = torch.stack([output[key] for output in output_batch], 0)
                metrics[key] = torch.mean(metrics[key], 0)
        metrics_data.append(metrics)
    # Return metrics data
    return metrics_data

def create_dict(metrics_data, metrics_keys, metrics_modes,
                dataset, name='depth'):
    """
    Creates a dictionary from collated metrics

    Parameters
    ----------
    metrics_data : list
        List containing collated metrics
    metrics_keys : list
        List of keys for the metrics
    metrics_modes
        List of modes for the metrics
    dataset : CfgNode
        Dataset configuration file
    name : str
        Name of the task for the metric

    Returns
    -------
    metrics_dict : dict
        Metrics dictionary
    """
    # Create metrics dictionary
    metrics_dict = {}
    # For all datasets
    for n, metrics in enumerate(metrics_data):
        if metrics: # If there are calculated metrics
            prefix = prepare_dataset_prefix(dataset, n)
            # For all keys
            for i, key in enumerate(metrics_keys):
                for mode in metrics_modes:
                    metrics_dict['{}-{}{}'.format(prefix, key, mode)] =\
                        metrics['{}{}'.format(name, mode)][i].item()
    # Return metrics dictionary
    return metrics_dict

########################################################################################################################

def average_key(batch_list, key):
    """
    Average key in a list of batches

    Parameters
    ----------
    batch_list : list of dict
        List containing dictionaries with the same keys
    key : str
        Key to be averaged

    Returns
    -------
    average : float
        Average of the value contained in key for all batches
    """
    values = [batch[key] for batch in batch_list]
    return sum(values) / len(values)

def average_sub_key(batch_list, key, sub_key):
    """
    Average subkey in a dictionary in a list of batches

    Parameters
    ----------
    batch_list : list of dict
        List containing dictionaries with the same keys
    key : str
        Key to be averaged
    sub_key :
        Sub key to be averaged (belonging to key)

    Returns
    -------
    average : float
        Average of the value contained in the sub_key of key for all batches
    """
    values = [batch[key][sub_key] for batch in batch_list]
    return sum(values) / len(values)

def average_loss_and_metrics(batch_list, prefix):
    """
    Average loss and metrics values in a list of batches

    Parameters
    ----------
    batch_list : list of dict
        List containing dictionaries with the same keys
    prefix : str
        Prefix string for metrics logging

    Returns
    -------
    values : dict
        Dictionary containing a 'loss' float entry and a 'metrics' dict entry
    """
    values = OrderedDict()
    key = 'loss'
    values['{}-{}'.format(prefix, key)] = \
        average_key(batch_list, key)
    key = 'metrics'
    for sub_key in batch_list[0][key].keys():
        values['{}-{}'.format(prefix, sub_key)] = \
            average_sub_key(batch_list, key, sub_key)
    return values

########################################################################################################################
