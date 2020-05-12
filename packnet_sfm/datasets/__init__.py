"""
PackNet-SfM datasets
====================

These datasets output images, camera calibration, depth maps and poses for depth and pose estimation

- KITTIDataset: reads from KITTI_raw
- DGPDataset: reads from a DGP .json file
- ImageDataset: reads from a folder containing image sequences (no support for depth maps)

"""

from packnet_sfm.datasets.kitti_dataset import KITTIDataset
from packnet_sfm.datasets.dgp_dataset import DGPDataset
from packnet_sfm.datasets.image_dataset import ImageDataset

__all__ = [
    "KITTIDataset",
    "DGPDataset",
    "ImageDataset",
]
