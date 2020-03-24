# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
Geometry utilities
"""

import numpy as np


def invert_pose_numpy(T):
    """
    'Invert' 4x4 extrinsic matrix

    Parameters
    ----------
    T: 4x4 matrix (world to camera)

    Returns
    -------
    4x4 matrix (camera to world)
    """
    Tc = np.copy(T)
    R, t = Tc[:3, :3], Tc[:3, 3]
    Tc[:3, :3], Tc[:3, 3] = R.T, - np.matmul(R.T, t)
    return Tc

