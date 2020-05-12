"""Provides helper methods for loading and parsing KITTI data."""

from collections import namedtuple

import numpy as np

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')


def rotx(t):
    """
    Rotation about the x-axis

    Parameters
    ----------
    t : float
        Theta angle

    Returns
    -------
    matrix : np.array [3,3]
        Rotation matrix
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """
    Rotation about the y-axis

    Parameters
    ----------
    t : float
        Theta angle

    Returns
    -------
    matrix : np.array [3,3]
        Rotation matrix
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """
    Rotation about the z-axis

    Parameters
    ----------
    t : float
        Theta angle

    Returns
    -------
    matrix : np.array [3,3]
        Rotation matrix
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    """
    Transformation matrix from rotation matrix and translation vector.

    Parameters
    ----------
    R : np.array [3,3]
        Rotation matrix
    t : np.array [3]
        translation vector

    Returns
    -------
    matrix : np.array [4,4]
        Transformation matrix
    """
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary

    Parameters
    ----------
    filepath : str
        File path to read from

    Returns
    -------
    calib : dict
        Dictionary with calibration values
    """
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def pose_from_oxts_packet(raw_data, scale):
    """
    Helper method to compute a SE(3) pose matrix from an OXTS packet

    Parameters
    ----------
    raw_data : dict
        Oxts data to read from
    scale : float
        Oxts scale

    Returns
    -------
    R : np.array [3,3]
        Rotation matrix
    t : np.array [3]
        Translation vector
    """
    packet = OxtsPacket(*raw_data)
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t


def load_oxts_packets_and_poses(oxts_files):
    """
    Generator to read OXTS ground truth data.
    Poses are given in an East-North-Up coordinate system
    whose origin is the first GPS position.

    Parameters
    ----------
    oxts_files : list of str
        List of oxts files to read from

    Returns
    -------
    oxts : list of dict
        List of oxts ground-truth data
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None

    oxts = []

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                packet = OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)

                R, t = pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = t

                T_w_imu = transform_from_rot_trans(R, t - origin)

                oxts.append(OxtsData(packet, T_w_imu))

    return oxts


