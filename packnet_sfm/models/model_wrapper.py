# Copyright 2020 Toyota Research Institute.  All rights reserved.

from collections import OrderedDict
import os
import time
import random
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from packnet_sfm.datasets.transforms import get_transforms
from packnet_sfm.utils.depth import inv2depth, post_process_inv_depth, compute_depth_metrics
from packnet_sfm.utils.horovod import print0, world_size, rank, on_rank_0
from packnet_sfm.utils.image import flip_lr
from packnet_sfm.utils.load import load_class, load_class_args_create, \
    load_network, filter_args
from packnet_sfm.utils.logging import pcolor
from packnet_sfm.utils.reduce import all_reduce_metrics, reduce_dict, \
    create_dict, average_loss_and_metrics
from packnet_sfm.utils.save import save_depth
from packnet_sfm.models.model_utils import stack_batch


class ModelWrapper(torch.nn.Module):
    """
    Top-level torch.nn.Module wrapper around a SfmModel (pose+depth networks).
    Designed to use models with high-level Trainer classes (cf. trainers/).

    Parameters
    ----------
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    """

    def __init__(self, config, resume=None, logger=None, load_datasets=True):
        super().__init__()

        # Store configuration, checkpoint and logger
        self.config = config
        self.logger = logger
        self.resume = resume

        # Set random seed
        set_random_seed(config.arch.seed)

        # Task metrics
        self.metrics_name = 'depth'
        self.metrics_keys = ('abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')
        self.metrics_modes = ('', '_pp', '_gt', '_pp_gt')

        # Model, optimizers, schedulers and datasets are None for now
        self.model = self.optimizer = self.scheduler = None
        self.train_dataset = self.validation_dataset = self.test_dataset = None
        self.current_epoch = 0

        # Prepare model
        self.prepare_model(resume)

        # Prepare datasets
        if load_datasets:
            # Requirements for validation (we only evaluate depth for now)
            validation_requirements = {'gt_depth': True, 'gt_pose': False}
            test_requirements = validation_requirements
            self.prepare_datasets(validation_requirements, test_requirements)

        # Preparations done
        self.config.prepared = True

    def prepare_model(self, resume=None):
        """Prepare self.model (incl. loading previous state)"""
        print0(pcolor('### Preparing Model', 'green'))
        self.model = setup_model(self.config.model, self.config.prepared)
        # Resume model if available
        if resume:
            print0(pcolor('### Resuming from {}'.format(
                resume['file']), 'magenta', attrs=['bold']))
            self.model = load_network(
                self.model, resume['state_dict'], 'model')
            if 'epoch' in resume:
                self.current_epoch = resume['epoch']

    def prepare_datasets(self, validation_requirements, test_requirements):
        """Prepare datasets for training, validation and test."""
        # Prepare datasets
        print0(pcolor('### Preparing Datasets', 'green'))

        augmentation = self.config.datasets.augmentation
        # Setup train dataset (requirements are given by the model itself)
        self.train_dataset = setup_dataset(
            self.config.datasets.train, 'train',
            self.model.train_requirements, **augmentation)
        # Setup validation dataset
        self.validation_dataset = setup_dataset(
            self.config.datasets.validation, 'validation',
            validation_requirements, **augmentation)
        # Setup test dataset
        self.test_dataset = setup_dataset(
            self.config.datasets.test, 'test',
            test_requirements, **augmentation)

    @property
    def depth_net(self):
        """Returns depth network."""
        return self.model.depth_net

    @property
    def pose_net(self):
        """Returns pose network."""
        return self.model.pose_net

    @property
    def logs(self):
        """Returns various logs for tracking."""
        params = OrderedDict()
        for param in self.optimizer.param_groups:
            params['{}_learning_rate'.format(param['name'].lower())] = param['lr']
        params['progress'] = self.progress
        return {
            **params,
            **self.model.logs,
        }

    @property
    def progress(self):
        """Returns training progress (current epoch / max. number of epochs)"""
        return self.current_epoch / self.config.arch.max_epochs

    def configure_optimizers(self):
        """Configure depth and pose optimizers and the corresponding scheduler."""

        params = []
        # Load optimizer
        optimizer = getattr(torch.optim, self.config.model.optimizer.name)
        # Depth optimizer
        if self.depth_net is not None:
            params.append({
                'name': 'Depth',
                'params': self.depth_net.parameters(),
                **filter_args(optimizer, self.config.model.optimizer.depth)
            })
        # Pose optimizer
        if self.pose_net is not None:
            params.append({
                'name': 'Pose',
                'params': self.pose_net.parameters(),
                **filter_args(optimizer, self.config.model.optimizer.pose)
            })
        # Create optimizer with parameters
        optimizer = optimizer(params)

        # Load and initialize scheduler
        scheduler = getattr(torch.optim.lr_scheduler, self.config.model.scheduler.name)
        scheduler = scheduler(optimizer, **filter_args(scheduler, self.config.model.scheduler))

        if self.resume:
            if 'optimizer' in self.resume:
                optimizer.load_state_dict(self.resume['optimizer'])
            if 'scheduler' in self.resume:
                scheduler.load_state_dict(self.resume['scheduler'])

        # Create class variables so we can use it internally
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Return optimizer and scheduler
        return optimizer, scheduler

    def train_dataloader(self):
        """Prepare training dataloader."""
        return setup_dataloader(self.train_dataset,
                                self.config.datasets.train, 'train')[0]

    def val_dataloader(self):
        """Prepare validation dataloader."""
        return setup_dataloader(self.validation_dataset,
                                self.config.datasets.validation, 'validation')

    def test_dataloader(self):
        """Prepare test dataloader."""
        return setup_dataloader(self.test_dataset,
                                self.config.datasets.test, 'test')

    def training_step(self, batch, *args):
        """Processes a training batch."""
        batch = stack_batch(batch)
        output = self.model(batch, progress=self.progress)
        return {
            'loss': output['loss'],
            'metrics': output['metrics']
        }

    def validation_step(self, batch, *args):
        """Processes a validation batch."""
        output = self.evaluate_depth(batch)
        if self.logger:
            self.logger.log_depth('val', batch, output, args,
                                  self.validation_dataset, world_size(),
                                  self.config.datasets.validation)
        return {
            'idx': batch['idx'],
            **output['metrics'],
        }

    def test_step(self, batch, *args):
        """Processes a test batch."""
        output = self.evaluate_depth(batch)
        save_depth(batch, output, args,
                   self.config.datasets.test,
                   self.config.save)
        return {
            'idx': batch['idx'],
            **output['metrics'],
        }

    def training_epoch_end(self, output_batch):
        """Finishes a training epoch."""

        # Calculate and reduce average loss and metrics per GPU
        loss_and_metrics = average_loss_and_metrics(output_batch, 'avg_train')
        loss_and_metrics = reduce_dict(loss_and_metrics, to_item=True)

        # Log to wandb
        if self.logger:
            self.logger.log_metrics({
                **self.logs, **loss_and_metrics,
            })

        return {
            **loss_and_metrics
        }

    def validation_epoch_end(self, output_data_batch):
        """Finishes a validation epoch."""

        # Reduce depth metrics
        metrics_data = all_reduce_metrics(
            output_data_batch, self.validation_dataset, self.metrics_name)

        # Create depth dictionary
        metrics_dict = create_dict(
            metrics_data, self.metrics_keys, self.metrics_modes,
            self.config.datasets.validation)

        # Print stuff
        self.print_metrics(metrics_data, self.config.datasets.validation)

        # Log to wandb
        if self.logger:
            self.logger.log_metrics({
                **metrics_dict, 'global_step': self.current_epoch + 1,
            })

        return {
            **metrics_dict
        }

    def test_epoch_end(self, output_data_batch):
        """Finishes a test epoch."""

        # Reduce depth metrics
        metrics_data = all_reduce_metrics(
            output_data_batch, self.test_dataset, self.metrics_name)

        # Create depth dictionary
        metrics_dict = create_dict(
            metrics_data, self.metrics_keys, self.metrics_modes,
            self.config.datasets.test)

        # Print stuff
        self.print_metrics(metrics_data, self.config.datasets.test)

        return {
            **metrics_dict
        }

    def forward(self, *args, **kwargs):
        """Runs the model and returns the output."""
        assert self.model is not None, 'Model not defined'
        return self.model(*args, **kwargs)

    def depth(self, *args, **kwargs):
        """Runs the pose network and returns the output."""
        assert self.depth_net is not None, 'Depth network not defined'
        return self.depth_net(*args, **kwargs)

    def pose(self, *args, **kwargs):
        """Runs the depth network and returns the output."""
        assert self.pose_net is not None, 'Pose network not defined'
        return self.pose_net(*args, **kwargs)

    def evaluate_depth(self, batch):
        """Evaluate batch to produce depth metrics."""
        # Get predicted depth
        inv_depths = self.model(batch)['inv_depths']
        depth = inv2depth(inv_depths[0])
        # Post-process predicted depth
        batch['rgb'] = flip_lr(batch['rgb'])
        if 'input_depth' in batch:
            batch['input_depth'] = flip_lr(batch['input_depth'])
        inv_depths_flipped = self.model(batch)['inv_depths']
        inv_depth_pp = post_process_inv_depth(
            inv_depths[0], inv_depths_flipped[0], method='mean')
        depth_pp = inv2depth(inv_depth_pp)
        batch['rgb'] = flip_lr(batch['rgb'])
        # Calculate predicted metrics
        metrics = OrderedDict()
        if 'depth' in batch:
            for mode in self.metrics_modes:
                metrics[self.metrics_name + mode] = compute_depth_metrics(
                    self.config.model.params, gt=batch['depth'],
                    pred=depth_pp if 'pp' in mode else depth,
                    use_gt_scale='gt' in mode)
        # Return metrics and extra information
        return {
            'metrics': metrics,
            'inv_depth': inv_depth_pp
        }

    @on_rank_0
    def print_metrics(self, metrics_data, dataset):
        """Print depth metrics on rank 0 if available"""
        if not metrics_data[0]:
            return

        hor_line = '|{:<}|'.format('*' * 93)
        met_line = '| {:^14} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} |'
        num_line = '{:<14} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f}'

        def wrap(string):
            return '| {} |'.format(string)

        print()
        print()
        print()
        print(hor_line)

        if self.optimizer is not None:
            bs = 'E: {} BS: {}'.format(self.current_epoch + 1,
                                       self.config.datasets.train.batch_size)
            if self.model is not None:
                bs += ' - {}'.format(self.config.model.name)
            lr = 'LR ({}):'.format(self.config.model.optimizer.name)
            for param in self.optimizer.param_groups:
                lr += ' {} {:.2e}'.format(param['name'], param['lr'])
            par_line = wrap(pcolor('{:<40}{:>51}'.format(bs, lr),
                                   'green', attrs=['bold', 'dark']))
            print(par_line)
            print(hor_line)

        print(met_line.format(*(('METRIC',) + self.metrics_keys)))
        for n, metrics in enumerate(metrics_data):
            print(hor_line)
            path_line = '{}'.format(
                os.path.join(dataset.path[n], dataset.split[n]))
            if len(dataset.cameras[n]) == 1: # only allows single cameras
                path_line += ' ({})'.format(dataset.cameras[n][0])
            print(wrap(pcolor('*** {:<87}'.format(path_line), 'magenta', attrs=['bold'])))
            print(hor_line)
            for key, metric in metrics.items():
                if self.metrics_name in key:
                    print(wrap(pcolor(num_line.format(
                        *((key.upper(),) + tuple(metric.tolist()))), 'cyan')))
        print(hor_line)

        if self.logger:
            run_line = wrap(pcolor('{:<60}{:>31}'.format(
                self.config.wandb.url, self.config.wandb.name), 'yellow', attrs=['dark']))
            print(run_line)
            print(hor_line)

        print()


def set_random_seed(seed):
    if seed >= 0:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_depth_net(config, prepared, **kwargs):
    """
    Create a depth network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    prepared : bool
        True if the network has been prepared before
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    depth_net : nn.Module
        Create depth network
    """
    print0(pcolor('DepthNet: %s' % config.name, 'yellow'))
    depth_net = load_class_args_create(config.name,
        paths=['packnet_sfm.networks.depth',],
        args={**config, **kwargs},
    )
    if not prepared and config.checkpoint_path is not '':
        depth_net = load_network(depth_net, config.checkpoint_path,
                                 ['depth_net', 'disp_network'])
    return depth_net


def setup_pose_net(config, prepared, **kwargs):
    """
    Create a pose network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    prepared : bool
        True if the network has been prepared before
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    pose_net : nn.Module
        Created pose network
    """
    print0(pcolor('PoseNet: %s' % config.name, 'yellow'))
    pose_net = load_class_args_create(config.name,
        paths=['packnet_sfm.networks.pose',],
        args={**config, **kwargs},
    )
    if not prepared and config.checkpoint_path is not '':
        pose_net = load_network(pose_net, config.checkpoint_path,
                                ['pose_net', 'pose_network'])
    return pose_net


def setup_model(config, prepared, **kwargs):
    """
    Create a model

    Parameters
    ----------
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    prepared : bool
        True if the model has been prepared before
    kwargs : dict
        Extra parameters for the model

    Returns
    -------
    model : nn.Module
        Created model
    """
    print0(pcolor('Model: %s' % config.name, 'yellow'))
    model = load_class(config.name, paths=['packnet_sfm.models',])(
        **{**config.loss, **kwargs})
    # Add depth network if required
    if 'depth_net' in model.network_requirements:
        model.add_depth_net(setup_depth_net(config.depth_net, prepared))
    # Add pose network if required
    if 'pose_net' in model.network_requirements:
        model.add_pose_net(setup_pose_net(config.pose_net, prepared))
    # If a checkpoint is provided, load pretrained model
    if not prepared and config.checkpoint_path is not '':
        model = load_network(model, config.checkpoint_path, 'model')
    # Return model
    return model


def setup_dataset(config, mode, requirements, **kwargs):
    """
    Create a dataset class

    Parameters
    ----------
    config : CfgNode
        Configuration (cf. configs/default_config.py)
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the dataset
    requirements : dict (string -> bool)
        Different requirements for dataset loading (gt_depth, gt_pose, etc)
    kwargs : dict
        Extra parameters for dataset creation

    Returns
    -------
    dataset : Dataset
        Dataset class for that mode
    """
    # If no dataset is given, return None
    if len(config.path) == 0:
        return None

    print0(pcolor('###### Setup %s datasets' % mode, 'red'))

    # Global shared dataset arguments
    dataset_args = {
        'back_context': config.back_context,
        'forward_context': config.forward_context,
        'data_transform': get_transforms(mode, **kwargs)
    }

    # Loop over all datasets
    datasets = []
    for i in range(len(config.split)):
        path_split = os.path.join(config.path[i], config.split[i])

        # Individual shared dataset arguments
        dataset_args_i = {
            'depth_type': config.depth_type[i] if 'gt_depth' in requirements else None,
            'input_depth_type': config.input_depth_type[i] if 'gt_depth' in requirements else None,
            'with_pose': 'gt_pose' in requirements,
        }

        # KITTI dataset
        if config.dataset[i] == 'KITTI':
            from packnet_sfm.datasets.kitti_dataset import KITTIDataset
            dataset = KITTIDataset(
                config.path[i], path_split,
                **dataset_args, **dataset_args_i,
            )
        # DGP dataset
        elif config.dataset[i] == 'DGP':
            from packnet_sfm.datasets.dgp_dataset import DGPDataset
            dataset = DGPDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
                cameras=config.cameras[i],
            )
        # Image dataset
        elif config.dataset[i] == 'Image':
            from packnet_sfm.datasets.image_dataset import ImageDataset
            dataset = ImageDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )
        else:
            ValueError('Unknown dataset %d' % config.dataset[i])

        # Repeat if needed
        if 'repeat' in config and config.repeat[i] > 1:
            dataset = ConcatDataset([dataset for _ in range(config.repeat[i])])
        datasets.append(dataset)

        # Display dataset information
        bar = '######### {:>7}'.format(len(dataset))
        if 'repeat' in config:
            bar += ' (x{})'.format(config.repeat[i])
        bar += ': {:<}'.format(path_split)
        print0(pcolor(bar, 'yellow'))

    # If training, concatenate all datasets into a single one
    if mode == 'train':
        datasets = [ConcatDataset(datasets)]

    return datasets


def worker_init_fn(worker_id):
    """Function to initialize workers"""
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)


def get_datasampler(dataset, mode):
    """Distributed data sampler"""
    return torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=(mode=='train'),
        num_replicas=world_size(), rank=rank())


def setup_dataloader(datasets, config, mode):
    """
    Create a dataloader class

    Parameters
    ----------
    datasets : list of Dataset
        List of datasets from which to create dataloaders
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the dataloader

    Returns
    -------
    dataloaders : list of Dataloader
        List of created dataloaders for each input dataset
    """
    return [(DataLoader(dataset,
                        batch_size=config.batch_size, shuffle=False,
                        pin_memory=True, num_workers=config.num_workers,
                        worker_init_fn=worker_init_fn,
                        sampler=get_datasampler(dataset, mode))
             ) for dataset in datasets]
