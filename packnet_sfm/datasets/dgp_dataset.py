# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
from packnet_sfm.utils.misc import make_list
from packnet_sfm.utils.types import is_tensor
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def stack_sample(sample):
    """Stack a sample from multiple sensors"""
    # If there is only one sensor don't do anything
    if len(sample) == 1:
        return sample[0]

    # Otherwise, stack sample
    stacked_sample = {}
    for key in sample[0]:
        # Global keys (do not stack)
        if key in ['idx', 'dataset_idx', 'sensor_name', 'filename']:
            stacked_sample[key] = sample[0][key]
        else:
            # Stack torch tensors
            if is_tensor(sample[0][key]):
                stacked_sample[key] = torch.cat([s[key].unsqueeze(0) for s in sample], 0)
    # Return stacked sample
    return stacked_sample

########################################################################################################################
#### DATASET
########################################################################################################################

class DGPDataset:
    """
    DGP dataset class

    Parameters
    ----------
    path : str
        Path to the dataset
    split : str {'train', 'val', 'test'}
        Which dataset split to use
    cameras : list of str
        Which cameras to get information from
    depth_type : str
        Which lidar will be used to generate ground-truth information
    with_pose : bool
        If enabled pose estimates are also returned
    with_semantic : bool
        If enabled semantic estimates are also returned
    back_context : int
        Size of the backward context
    forward_context : int
        Size of the forward context
    data_transform : Function
        Transformations applied to the sample
    """
    def __init__(self, path, split,
                 cameras=None,
                 depth_type=None,
                 with_pose=False,
                 with_semantic=False,
                 back_context=0,
                 forward_context=0,
                 data_transform=None,
                 ):
        self.split = split
        self.dataset_idx = 0

        self.bwd = back_context
        self.fwd = forward_context
        self.has_context = back_context + forward_context > 0

        self.num_cameras = len(cameras)
        self.data_transform = data_transform

        self.with_depth = depth_type is not None
        self.with_pose = with_pose
        self.with_semantic = with_semantic

        self.dataset = SynchronizedSceneDataset(path,
            split=split,
            datum_names=cameras,
            backward_context=back_context,
            forward_context=forward_context,
            generate_depth_from_datum=depth_type,
            requested_annotations=None,
            only_annotated_datums=False,
        )

    def get_current(self, key, sensor_idx):
        """Return current timestep of a key from a sensor"""
        return self.sample_dgp[self.bwd][sensor_idx][key]

    def get_backward(self, key, sensor_idx):
        """Return backward timesteps of a key from a sensor"""
        return [] if self.bwd == 0 else \
            [self.sample_dgp[i][sensor_idx][key] \
             for i in range(0, self.bwd)]

    def get_forward(self, key, sensor_idx):
        """Return forward timestep of a key from a sensor"""
        return [] if self.fwd == 0 else \
            [self.sample_dgp[i][sensor_idx][key] \
             for i in range(self.bwd + 1, self.bwd + self.fwd + 1)]

    def get_context(self, key, sensor_idx):
        """Get both backward and forward contexts"""
        return self.get_backward(key, sensor_idx) + self.get_forward(key, sensor_idx)

    def __len__(self):
        """Length of dataset"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a dataset sample"""
        # Get DGP sample (if single sensor, make it a list)
        self.sample_dgp = self.dataset[idx]
        self.sample_dgp = [make_list(sample) for sample in self.sample_dgp]

        # Loop over all cameras
        sample = []
        for i in range(self.num_cameras):
            data = {
                'idx': idx,
                'dataset_idx': self.dataset_idx,
                'sensor_name': self.get_current('datum_name', i),
                'filename': '%s_%010d' % (self.split, idx),
                #
                'rgb': self.get_current('rgb', i),
                'intrinsics': self.get_current('intrinsics', i),
            }

            if self.with_depth:
                data.update({
                    'depth': self.get_current('depth', i),
                })

            if self.with_pose:
                data.update({
                    'extrinsics': [pose.matrix for pose in self.get_current('extrinsics', i)],
                    'pose': [pose.matrix for pose in self.get_current('pose', i)],
                })

            if self.has_context:
                data.update({
                    'rgb_context': self.get_context('rgb', i),
                })

            sample.append(data)

        # Apply same data transformations for all sensors
        if self.data_transform:
            sample = [self.data_transform(smp) for smp in sample]

        # Return sample (stacked if necessary)
        return stack_sample(sample)

########################################################################################################################

