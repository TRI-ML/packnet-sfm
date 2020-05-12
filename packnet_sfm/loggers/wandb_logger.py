# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from Pytorch-Lightning
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/wandb.py

from argparse import Namespace
from collections import OrderedDict
import numpy as np
import torch.nn as nn
import wandb
from wandb.wandb_run import Run

from packnet_sfm.utils.depth import viz_inv_depth
from packnet_sfm.utils.logging import prepare_dataset_prefix
from packnet_sfm.utils.types import is_dict, is_tensor


class WandbLogger:
    """
    Wandb logger class to monitor training.

    Parameters
    ----------
    name : str
        Run name (if empty, uses a fancy Wandb name, highly recommended)
    dir : str
        Folder where wandb information is stored
    id : str
        ID for the run
    anonymous : bool
        Anonymous mode
    version : str
        Run version
    project : str
        Wandb project where the run will live
    tags : list of str
        List of tags to append to the run
    log_model : bool
        Log the model to wandb or not
    experiment : wandb
        Wandb experiment
    entity : str
        Wandb entity
    """
    def __init__(self,
                 name=None, dir=None, id=None, anonymous=False,
                 version=None, project=None, entity=None,
                 tags=None, log_model=False, experiment=None
                 ):
        super().__init__()
        self._name = name
        self._dir = dir
        self._anonymous = 'allow' if anonymous else None
        self._id = version or id
        self._tags = tags
        self._project = project
        self._entity = entity
        self._log_model = log_model

        self._experiment = experiment if experiment else self.create_experiment()
        self._metrics = OrderedDict()

    def __getstate__(self):
        """Get the current logger state"""
        state = self.__dict__.copy()
        state['_id'] = self._experiment.id if self._experiment is not None else None
        state['_experiment'] = None
        return state

    def create_experiment(self):
        """Creates and returns a new experiment"""
        experiment = wandb.init(
            name=self._name, dir=self._dir, project=self._project,
            anonymous=self._anonymous, reinit=True, id=self._id,
            resume='allow', tags=self._tags, entity=self._entity
        )
        wandb.run.save()
        return experiment

    def watch(self, model: nn.Module, log: str = 'gradients', log_freq: int = 100):
        """Watch training parameters."""
        self.experiment.watch(model, log=log, log_freq=log_freq)

    @property
    def experiment(self) -> Run:
        """Returns the experiment (creates a new if it doesn't exist)."""
        if self._experiment is None:
            self._experiment = self.create_experiment()
        return self._experiment

    @property
    def version(self) -> str:
        """Returns experiment version."""
        return self._experiment.id if self._experiment else None

    @property
    def name(self) -> str:
        """Returns experiment name."""
        name = self._experiment.project_name() if self._experiment else None
        return name

    @property
    def run_name(self) -> str:
        """Returns run name."""
        return wandb.run.name if self._experiment else None

    @property
    def run_url(self) -> str:
        """Returns run URL."""
        return 'https://app.wandb.ai/{}/{}/runs/{}'.format(
            wandb.run.entity, wandb.run.project, wandb.run.id) if self._experiment else None

    @staticmethod
    def _convert_params(params):
        if isinstance(params, Namespace):
            params = vars(params)
        if params is None:
            params = {}
        return params

    def log_config(self, params):
        """Logs model configuration."""
        params = self._convert_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics):
        """Logs training metrics."""
        self._metrics.update(metrics)
        if 'global_step' in metrics:
            self.experiment.log(self._metrics)
            self._metrics.clear()

    def log_images(self, func, mode, batch, output,
                   args, dataset, world_size, config):
        """
        Adds images to metrics for later logging.

        Parameters
        ----------
        func : Function
            Function used to process the image before logging
        mode : str {"train", "val"}
            Training stage where the images come from (serve as prefix for logging)
        batch : dict
            Data batch
        output : dict
            Model output
        args : tuple
            Step arguments
        dataset : CfgNode
            Dataset configuration
        world_size : int
            Number of GPUs, used to get logging samples at consistent intervals
        config : CfgNode
            Model configuration
        """
        dataset_idx = 0 if len(args) == 1 else args[1]
        prefix = prepare_dataset_prefix(config, dataset_idx)
        interval = len(dataset[dataset_idx]) // world_size // config.num_logs
        if args[0] % interval == 0:
            prefix_idx = '{}-{}-{}'.format(mode, prefix, batch['idx'][0].item())
            func(prefix_idx, batch, output)

    # Log depth images
    def log_depth(self, *args, **kwargs):
        """Helper function used to log images relevant for depth estimation"""
        def log(prefix_idx, batch, output):
            self._metrics.update(log_rgb('rgb', prefix_idx, batch))
            self._metrics.update(log_inv_depth('inv_depth', prefix_idx, output))
            if 'depth' in batch:
                self._metrics.update(log_depth('depth', prefix_idx, batch))
        self.log_images(log, *args, **kwargs)


def log_rgb(key, prefix, batch, i=0):
    """
    Converts an RGB image from a batch for logging

    Parameters
    ----------
    key : str
        Key from data containing the image
    prefix : str
        Prefix added to the key for logging
    batch : dict
        Dictionary containing the key
    i : int
        Batch index from which to get the image

    Returns
    -------
    image : wandb.Image
        Wandb image ready for logging
    """
    rgb = batch[key] if is_dict(batch) else batch
    return prep_image(prefix, key,
                      rgb[i])


def log_depth(key, prefix, batch, i=0):
    """
    Converts a depth map from a batch for logging

    Parameters
    ----------
    key : str
        Key from data containing the depth map
    prefix : str
        Prefix added to the key for logging
    batch : dict
        Dictionary containing the key
    i : int
        Batch index from which to get the depth map

    Returns
    -------
    image : wandb.Image
        Wandb image ready for logging
    """
    depth = batch[key] if is_dict(batch) else batch
    inv_depth = 1. / depth[i]
    inv_depth[depth[i] == 0] = 0
    return prep_image(prefix, key,
                      viz_inv_depth(inv_depth, filter_zeros=True))


def log_inv_depth(key, prefix, batch, i=0):
    """
    Converts an inverse depth map from a batch for logging

    Parameters
    ----------
    key : str
        Key from data containing the inverse depth map
    prefix : str
        Prefix added to the key for logging
    batch : dict
        Dictionary containing the key
    i : int
        Batch index from which to get the inverse depth map

    Returns
    -------
    image : wandb.Image
        Wandb image ready for logging
    """
    inv_depth = batch[key] if is_dict(batch) else batch
    return prep_image(prefix, key,
                      viz_inv_depth(inv_depth[i]))


def prep_image(prefix, key, image):
    """
    Prepare image for wandb logging

    Parameters
    ----------
    prefix : str
        Prefix added to the key for logging
    key : str
        Key from data containing the inverse depth map
    image : torch.Tensor [3,H,W]
        Image to be logged

    Returns
    -------
    output : dict
        Dictionary with key and value for logging
    """
    if is_tensor(image):
        image = image.detach().permute(1, 2, 0).cpu().numpy()
    prefix_key = '{}-{}'.format(prefix, key)
    return {prefix_key: wandb.Image(image, caption=key)}
