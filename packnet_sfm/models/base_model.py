# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base Model class defines APIs for packnet_sfm model wrapper.

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """
    def __init__(self, **kwargs):
        super().__init__()

        self._logs = {}
        self._losses = {}

        self._network_requirements = []     # Which networks the model requires
        self._train_requirements = []       # Which GT information the model requires at training time
        self._input_keys = ['rgb']          # Which input keys are provided to the model

    def _forward_unimplemented(self, *args):
        pass

    @property
    def logs(self):
        """Return logs."""
        return self._logs

    @property
    def losses(self):
        """Return metrics."""
        return self._losses

    def add_loss(self, key, val):
        """Add a new loss to the dictionary and detaches it."""
        self._losses[key] = val.detach()

    @property
    def network_requirements(self):
        """
        Networks required to run the model

        Returns
        -------
        requirements : dict
            key : str
                Attribute name in model object pointing to corresponding network.
            value : str
                Task Name.
        """
        return self._network_requirements

    @property
    def train_requirements(self):
        """
        Information required by the model at training stage

        Returns
        -------
        requirements : dict
            gt_depth : bool
                Whether ground truth depth is required by the model at training time
            gt_pose : bool
                Whether ground truth pose is required by the model at training time
        """
        return self._train_requirements

    def add_net(self, network_module, network_name):
        """Add a network module as an attribute in the model

        Parameters
        ----------
        network_module: torch.nn.Module

        network_name: str
            name of the network as well as the attribute in the network.
        """
        assert network_name in self._network_requirements, "Network module not required!"
        setattr(self, network_name, network_module)

    def forward(self, batch, return_logs=False, **kwargs):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        """
        raise NotImplementedError("Please implement forward function in your own subclass model.")
