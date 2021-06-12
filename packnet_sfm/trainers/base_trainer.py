# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
from tqdm import tqdm
from packnet_sfm.utils.logging import prepare_dataset_prefix


def sample_to_cuda(data, dtype=None):
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        return {key: sample_to_cuda(data[key], dtype) for key in data.keys()}
    elif isinstance(data, list):
        return [sample_to_cuda(val, dtype) for val in data]
    else:
        # only convert floats (e.g., to half), otherwise preserve (e.g, ints)
        dtype = dtype if torch.is_floating_point(data) else None
        return data.to('cuda', dtype=dtype)


class BaseTrainer:
    def __init__(self, min_epochs=0, max_epochs=50,
                 validate_first=False, checkpoint=None, **kwargs):

        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.validate_first = validate_first

        self.checkpoint = checkpoint
        self.module = None

    @property
    def proc_rank(self):
        raise NotImplementedError('Not implemented for BaseTrainer')

    @property
    def world_size(self):
        raise NotImplementedError('Not implemented for BaseTrainer')

    @property
    def is_rank_0(self):
        return self.proc_rank == 0

    def check_and_save(self, module, output):
        if self.checkpoint:
            self.checkpoint.check_and_save(module, output)

    def train_progress_bar(self, dataloader, config, ncols=120):
        return tqdm(enumerate(dataloader, 0),
                    unit=' images', unit_scale=self.world_size * config.batch_size,
                    total=len(dataloader), smoothing=0,
                    disable=not self.is_rank_0, ncols=ncols,
                    )

    def val_progress_bar(self, dataloader, config, n=0, ncols=120):
        return tqdm(enumerate(dataloader, 0),
                    unit=' images', unit_scale=self.world_size * config.batch_size,
                    total=len(dataloader), smoothing=0,
                    disable=not self.is_rank_0, ncols=ncols,
                    desc=prepare_dataset_prefix(config, n)
                    )

    def test_progress_bar(self, dataloader, config, n=0, ncols=120):
        return tqdm(enumerate(dataloader, 0),
                    unit=' images', unit_scale=self.world_size * config.batch_size,
                    total=len(dataloader), smoothing=0,
                    disable=not self.is_rank_0, ncols=ncols,
                    desc=prepare_dataset_prefix(config, n)
                    )
