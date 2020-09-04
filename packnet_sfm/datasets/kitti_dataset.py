# Copyright 2020 Toyota Research Institute.  All rights reserved.

import glob
import numpy as np
import os
import packnet_sfm.datasets.sintel_io as sio

from torch.utils.data import Dataset

from packnet_sfm.datasets.kitti_dataset_utils import \
    pose_from_oxts_packet, read_calib_file, transform_from_rot_trans
from packnet_sfm.utils.image import load_image
from packnet_sfm.geometry.pose_utils import invert_pose_numpy

########################################################################################################################

def dummy_calibration(seq_name='alley_1'):
    #return np.array([[688.00006104,   0.,         511.5       ],
    #                 [  0.,         688.00006104, 217.5       ],
    #                 [  0.,           0.,           1.        ]])
    #return np.array([[1.120e+03, 0.000e+00, 5.115e+02],
    #                 [0.000e+00, 1.120e+03, 2.175e+02],
    #                 [0.000e+00, 0.000e+00, 1.000e+00]])
    #print('dummy - seq_name: ', seq_name)
    if seq_name=='alley_1':
        return np.array([[688.00006104,   0.,         511.5       ],
                         [  0.,         688.00006104, 191.5       ],
                         [  0.,           0.,           1.        ]])
    elif seq_name=='alley_2':
        return np.array([[576.,    0.,  511.5],
                         [  0.,  576.,  191.5],
                         [  0.,    0.,    1. ]])
    elif seq_name=='ambush_2':
        return np.array([[640.,    0.,  511.5],
                         [  0.,  640.,  191.5],
                         [  0.,    0.,    1. ]])
    elif seq_name=='ambush_4':
        return np.array([[1.120e+03, 0.000e+00, 5.115e+02],
                         [0.000e+00, 1.120e+03, 1.915e+02],
                         [0.000e+00, 0.000e+00, 1.000e+00]])
    elif seq_name=='ambush_5':
        return np.array([[1.120e+03, 0.000e+00, 5.115e+02],
                         [0.000e+00, 1.120e+03, 1.915e+02],
                         [0.000e+00, 0.000e+00, 1.000e+00]])
    elif seq_name=='ambush_6':
        return np.array([[576.,    0.,  511.5],
                         [  0.,  576.,  191.5],
                         [  0.,    0.,    1. ]])
    elif seq_name=='bamboo_1':
        return np.array([[800.,    0.,  511.5],
                         [  0.,  800.,  191.5],
                         [  0.,    0.,    1. ]])
    elif seq_name=='bamboo_2':
        return np.array([[640.,    0.,  511.5],
                         [  0.,  640.,  191.5],
                         [  0.,    0.,    1. ]])
    elif seq_name=='cave_2':
        return np.array([[1.52859412e+03, 0.00000000e+00, 5.11500000e+02],
                         [0.00000000e+00, 1.52859412e+03, 1.91500000e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    elif seq_name=='cave_4':
        return np.array([[640.,    0.,  511.5],
                         [  0.,  640.,  191.5],
                         [  0.,    0.,    1. ]])
    elif seq_name=='market_2':
        return np.array([[3.200e+03, 0.000e+00, 5.115e+02],
                         [0.000e+00, 3.200e+03, 1.915e+02],
                         [0.000e+00, 0.000e+00, 1.000e+00]])
    elif seq_name=='market_5':
        return np.array([[1.120e+03, 0.000e+00, 5.115e+02],
                         [0.000e+00, 1.120e+03, 1.915e+02],
                         [0.000e+00, 0.000e+00, 1.000e+00]])
    elif seq_name=='market_6':
        return np.array([[640.,    0.,  511.5],
                         [  0.,  640.,  191.5],
                         [  0.,    0.,    1. ]])
    elif seq_name=='mountain_1':
        return np.array([[1.120e+03, 0.000e+00, 5.115e+02],
                         [0.000e+00, 1.120e+03, 1.915e+02],
                         [0.000e+00, 0.000e+00, 1.000e+00]])
    elif seq_name=='shaman_2':
        return np.array([[1.600e+03, 0.000e+00, 5.115e+02],
                         [0.000e+00, 1.600e+03, 1.915e+02],
                         [0.000e+00, 0.000e+00, 1.000e+00]])
    elif seq_name=='shaman_3':
        return np.array([[1.120e+03, 0.000e+00, 5.115e+02],
                         [0.000e+00, 1.120e+03, 1.915e+02],
                         [0.000e+00, 0.000e+00, 1.000e+00]])
    elif seq_name=='sleeping_1':
        return np.array([[640.,    0.,  511.5],
                         [  0.,  640.,  191.5],
                         [  0.,    0.,    1. ]])
    elif seq_name=='sleeping_2':
        return np.array([[640.,    0.,  511.5],
                         [  0.,  640.,  191.5],
                         [  0.,    0.,    1. ]])
    else:
        return np.array([[640.,    0.,  511.5],
                         [  0.,  640.,  191.5],
                         [  0.,    0.,    1. ]])



# Cameras from the stero pair (left is the origin)
IMAGE_FOLDER = {
    'left': 'image_02',
    'right': 'image_03',
}
# Name of different calibration files
CALIB_FILE = {
    'cam2cam': 'calib_cam_to_cam.txt',
    'velo2cam': 'calib_velo_to_cam.txt',
    'imu2velo': 'calib_imu_to_velo.txt',
}
PNG_DEPTH_DATASETS = ['groundtruth']
OXTS_POSE_DATA = 'oxts'

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def read_npz_depth(file, depth_type):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)[depth_type + '_depth'].astype(np.float32)
    return np.expand_dims(depth, axis=2)

def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(load_image(file), dtype=int)
    assert (np.max(depth_png) > 255), 'Wrong .png depth file'
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return np.expand_dims(depth, axis=2)

########################################################################################################################
#### DATASET
########################################################################################################################

class KITTIDataset(Dataset):
    """
    KITTI dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    file_list : str
        Split file, with paths to the images to be used
    train : bool
        True if the dataset will be used for training
    data_transform : Function
        Transformations applied to the sample
    depth_type : str
        Which depth type to load
    with_pose : bool
        True if returning ground-truth pose
    back_context : int
        Number of backward frames to consider as context
    forward_context : int
        Number of forward frames to consider as context
    strides : tuple
        List of context strides
    """
    def __init__(self, root_dir, file_list, train=True,
                 data_transform=None, depth_type=None, with_pose=False,
                 back_context=0, forward_context=0, strides=(1,), seq_name='alley_1'):
        # Assertions
        backward_context = back_context
        assert backward_context >= 0 and forward_context >= 0, 'Invalid contexts'

        self.backward_context = backward_context
        self.backward_context_paths = []
        self.forward_context = forward_context
        self.forward_context_paths = []
        self.seq_name = seq_name

        self.with_context = (backward_context != 0 or forward_context != 0)
        self.split = file_list.split('/')[-1].split('.')[0]

        self.train = self.with_context
        if self.train:
            print('#'*30)
            print(backward_context)
            print(forward_context)
            #backward_context=1
            #forward_context=1
            #self.with_context = True
        self.root_dir = root_dir
        self.data_transform = data_transform

        self.depth_type = depth_type
        print('depth type: ', depth_type)
        self.with_depth = depth_type is not '' and depth_type is not None
        self.with_depth = False
        self.with_pose = with_pose

        self._cache = {}
        self.pose_cache = {}
        self.oxts_cache = {}
        self.imu2velo_calib_cache = {}
        self.sequence_origin_cache = {}

        with open(file_list, "r") as f:
            data = f.readlines()

        self.paths = []
        self.paths_depth = []
        # Get file list from data
        for i, fname in enumerate(data):
            self.seq_name = fname.split()[0][:-15]
            path = os.path.join(root_dir, fname.split()[0])
            if not self.with_depth:
                #print('valda without depth')
                self.paths.append(path)
                self.paths_depth.append(fname.split()[0])
            else:
                # Check if the depth file exists
                depth = self._get_depth_file(os.path.join(root_dir, 'depth', fname.split()[0]))
                depth = depth[:-4] + '.dpt'
                #print('depth: ', depth)
                if depth is not None and os.path.exists(depth):
                    #print('exists')
                    self.paths.append(path)
                    self.paths_depth.append(depth)
        print('paths length: ', len(self.paths))

        # If using context, filter file list
        #####buraya bak
        print('with context?: ' ,self.with_context)
        if self.with_context:
            print('with context: ')
            #print(self.paths)
            paths_with_context = []
            for stride in strides:
                #print('stride: ', stride)
                for idx, file in enumerate(self.paths):
                    #print(idx)
                    #print(file)
                    backward_context_idxs, forward_context_idxs = \
                        self._get_sample_context(
                            file, backward_context, forward_context, stride)
                    #print(backward_context_idxs)
                    #print(forward_context_idxs)
                    if backward_context_idxs is not None and forward_context_idxs is not None:
                        paths_with_context.append(self.paths[idx])
                        self.forward_context_paths.append(forward_context_idxs)
                        self.backward_context_paths.append(backward_context_idxs[::-1])
            self.paths = paths_with_context
            #print(paths_with_context)
        print(self.paths)
########################################################################################################################

    @staticmethod
    def _get_next_file(idx, file):
        """Get next file given next idx and current file."""
        #print('IN NEXT FILE')
        #print('idx: ', idx)
        #print('basename: ', os.path.basename(file))
        base, ext = os.path.splitext(os.path.basename(file))
        #print('base: ', base)
        #print('ext: ', ext)
        #print(('dirname: ', os.path.dirname(file)))
        #print(('lenbase: ',len(base)))
        #print(os.path.join(os.path.dirname(file), 'frame_' + str(idx).zfill(4) + ext))
        return os.path.join(os.path.dirname(file), 'frame_' + str(idx).zfill(4) + ext)

    @staticmethod
    def _get_parent_folder(image_file):
        """Get the parent folder from image_file."""
        return os.path.abspath(os.path.join(image_file, "../../../.."))

    @staticmethod
    def _get_intrinsics(seq_name):
        """Get intrinsics from the calib_data dictionary."""
        return dummy_calibration(seq_name)

    @staticmethod
    def _read_raw_calib_file(folder):
        """Read raw calibration files from folder."""
        return read_calib_file(os.path.join(folder, CALIB_FILE['cam2cam']))

########################################################################################################################
#### DEPTH
########################################################################################################################

    def _read_depth(self, depth_file):
        """Get the depth map from a file."""
        if self.depth_type in ['velodyne']:
            return read_npz_depth(depth_file, self.depth_type)
        elif self.depth_type in ['groundtruth']:
            #return read_png_depth(depth_file)
            np.expand_dims(sio.depth_read(depth_file),axis=2)
        else:
            raise NotImplementedError(
                'Depth type {} not implemented'.format(self.depth_type))

    def _get_depth_file(self, image_file):
        """Get the corresponding depth file from an image file."""
        depth_file=image_file
        return depth_file

    def _get_sample_context(self, sample_name,
                            backward_context, forward_context, stride=1):
        """
        Get a sample context

        Parameters
        ----------
        sample_name : str
            Path + Name of the sample
        backward_context : int
            Size of backward context
        forward_context : int
            Size of forward context
        stride : int
            Stride value to consider when building the context

        Returns
        -------
        backward_context : list of int
            List containing the indexes for the backward context
        forward_context : list of int
            List containing the indexes for the forward context
        """
        base, ext = os.path.splitext(os.path.basename(sample_name))
        #print('base: ', base)
        #print('ext: ', ext)
        parent_folder = os.path.dirname(sample_name)
        #print('parent_folder: ', parent_folder)
        f_idx = int(base[-4:])
        #print('f_idx: ',f_idx)

        # Check number of files in folder
        #print('self._cache:' , self._cache)
        if parent_folder in self._cache:
            max_num_files = self._cache[parent_folder]
            #print('max_num_files: ', max_num_files)
        else:
            max_num_files = len(glob.glob(os.path.join(parent_folder, '*' + ext)))
            #print('glob files')
            #print(glob.glob(os.path.join(parent_folder, '*' + ext)))
            self._cache[parent_folder] = max_num_files
            #print('max_num_files: ', max_num_files)

        # Check bounds
        #print('backward_context: ', backward_context)
        #print('forward: ', forward_context)
        #print('stride: ', stride)
        #print(str(f_idx - backward_context * stride))
        #print(str(f_idx + forward_context * stride))
        if (f_idx - backward_context * stride) < 0 or (
                f_idx + forward_context * stride) >= max_num_files:
            #print('quit')
            return None, None

        # Backward context
        c_idx = f_idx
        backward_context_idxs = []
        while len(backward_context_idxs) < backward_context and c_idx > 0:
            c_idx -= stride
            filename = self._get_next_file(c_idx, sample_name)
            #print('filename: ', filename)
            if os.path.exists(filename):
                backward_context_idxs.append(c_idx)
        if c_idx < 0:
            return None, None

        # Forward context
        c_idx = f_idx
        forward_context_idxs = []
        while len(forward_context_idxs) < forward_context and c_idx < max_num_files:
            c_idx += stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                forward_context_idxs.append(c_idx)
        if c_idx >= max_num_files:
            return None, None

        return backward_context_idxs, forward_context_idxs

    def _get_context_files(self, sample_name, idxs):
        """
        Returns image and depth context files

        Parameters
        ----------
        sample_name : str
            Name of current sample
        idxs : list of idxs
            Context indexes

        Returns
        -------
        image_context_paths : list of str
            List of image names for the context
        depth_context_paths : list of str
            List of depth names for the context
        """
        image_context_paths = [self._get_next_file(i, sample_name) for i in idxs]
        if self.with_depth:
            depth_context_paths = [self._get_depth_file(f) for f in image_context_paths]
            return image_context_paths, depth_context_paths
        else:
            return image_context_paths, None

########################################################################################################################
#### POSE
########################################################################################################################

    def _get_imu2cam_transform(self, image_file):
        """Gets the transformation between IMU an camera from an image file"""
        parent_folder = self._get_parent_folder(image_file)
        if image_file in self.imu2velo_calib_cache:
            return self.imu2velo_calib_cache[image_file]

        cam2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['cam2cam']))
        imu2velo = read_calib_file(os.path.join(parent_folder, CALIB_FILE['imu2velo']))
        velo2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['velo2cam']))

        velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
        cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

        imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat
        self.imu2velo_calib_cache[image_file] = imu2cam
        return imu2cam

    @staticmethod
    def _get_oxts_file(image_file):
        """Gets the oxts file from an image file."""
        # find oxts pose file
        for cam in ['left', 'right']:
            # Check for both cameras, if found replace and return file name
            if IMAGE_FOLDER[cam] in image_file:
                return image_file.replace(IMAGE_FOLDER[cam], OXTS_POSE_DATA).replace('.png', '.txt')
        # Something went wrong (invalid image file)
        raise ValueError('Invalid KITTI path for pose supervision.')

    def _get_oxts_data(self, image_file):
        """Gets the oxts data from an image file."""
        oxts_file = self._get_oxts_file(image_file)
        if oxts_file in self.oxts_cache:
            oxts_data = self.oxts_cache[oxts_file]
        else:
            oxts_data = np.loadtxt(oxts_file, delimiter=' ', skiprows=0)
            self.oxts_cache[oxts_file] = oxts_data
        return oxts_data

    def _get_pose(self, image_file):
        """Gets the pose information from an image file."""
        if image_file in self.pose_cache:
            return self.pose_cache[image_file]
        # Find origin frame in this sequence to determine scale & origin translation
        base, ext = os.path.splitext(os.path.basename(image_file))
        origin_frame = os.path.join(os.path.dirname(image_file), str(0).zfill(len(base)) + ext)
        # Get origin data
        origin_oxts_data = self._get_oxts_data(origin_frame)
        lat = origin_oxts_data[0]
        scale = np.cos(lat * np.pi / 180.)
        # Get origin pose
        origin_R, origin_t = pose_from_oxts_packet(origin_oxts_data, scale)
        origin_pose = transform_from_rot_trans(origin_R, origin_t)
        # Compute current pose
        oxts_data = self._get_oxts_data(image_file)
        R, t = pose_from_oxts_packet(oxts_data, scale)
        pose = transform_from_rot_trans(R, t)
        # Compute odometry pose
        imu2cam = self._get_imu2cam_transform(image_file)
        odo_pose = (imu2cam @ np.linalg.inv(origin_pose) @
                    pose @ np.linalg.inv(imu2cam)).astype(np.float32)
        # Cache and return pose
        self.pose_cache[image_file] = odo_pose
        return odo_pose

########################################################################################################################
    def center_crop_im(self, im, new_width=1024, new_height=384):
        width, height = im.size   # Get dimensions
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        # Crop the center of the image
        #return im
        im = im.crop((left, top, right, bottom))
        return im

    def __len__(self):
        """Dataset length."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Get dataset sample given an index."""
        # Add image information
        #print('LOADING IMAGE')
        #print(np.array(load_image(self.paths[idx])).shape)
        #print(np.array(self.center_crop_im(load_image(self.paths[idx]))).shape)
        sample = {
            'idx': idx,
            'filename': '%s_%010d' % (self.split, idx),
            'rgb': self.center_crop_im(load_image(self.paths[idx])),
        }

        # Add intrinsics
        parent_folder = self._get_parent_folder(self.paths[idx])
        sample.update({
            'intrinsics': self._get_intrinsics(self.seq_name),
        })

        # Add pose information if requested
        if self.with_pose:
            sample.update({
                'pose': self._get_pose(self.paths[idx]),
            })

        # Add depth information if requested
        if self.with_depth:
            sample.update({
                'depth': self._read_depth(self._get_depth_file(self.paths_depth[idx])),
            })

        # Add context information if requested
        if self.with_context:
            # Add context images
            all_context_idxs = self.backward_context_paths[idx] + \
                               self.forward_context_paths[idx]
            image_context_paths, _ = \
                self._get_context_files(self.paths[idx], all_context_idxs)
            image_context = [self.center_crop_im(load_image(f)) for f in image_context_paths]
            sample.update({
                'rgb_context': image_context
            })
            # Add context poses
            if self.with_pose:
                first_pose = sample['pose']
                image_context_pose = [self._get_pose(f) for f in image_context_paths]
                image_context_pose = [invert_pose_numpy(context_pose) @ first_pose
                                      for context_pose in image_context_pose]
                sample.update({
                    'pose_context': image_context_pose
                })

        # Apply transformations
        if self.data_transform:
            sample = self.data_transform(sample)

        # Return sample
        return sample

########################################################################################################################




