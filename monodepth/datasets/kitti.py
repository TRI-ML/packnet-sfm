# Copyright 2020 Toyota Research Institute.  All rights reserved.
# code adapted from https://github.com/ClubAI/MonoDepth-PyTorch/blob/master/data_loader.py

import logging
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from monodepth.externals.pykitti_utils import pose_from_oxts_packet, read_calib_file, \
    transform_from_rot_trans

LEFT_STEREO_IMAGE_FOLDER = 'image_02/data'
RIGHT_STEREO_IMAGE_FOLDER = 'image_03/data'
PNG_DATASET = ['groundtruth']                       # list of depth_types (datasets) that need to load *.pngs
CAMERA_CALIBRATION_FILE = 'calib_cam_to_cam.txt'


class KittiLoader(Dataset):
    def __init__(self, root_dir, file_list, data_transform=None, depth_type=None, append_left_and_right_images=False):

        assert (depth_type is None or depth_type in ['groundtruth', 'velodyne'])
        self.with_depth = depth_type is not None
        self.depth_type = depth_type
        self.data_transform = data_transform
        self.root_dir = root_dir
        self.append_left_and_right_images = append_left_and_right_images

        self.left_paths = []
        self.right_paths = []
        self.calibration_cache = {}

        with open(file_list, "r") as f:
            data = f.readlines()
        logging.info('Found {} files in split.'.format(len(data)))

        for i, fname in enumerate(data):
            # assuming these two files exist ;-)
            left = os.path.join(root_dir, fname.split()[0])
            right = os.path.join(root_dir, fname.split()[1])

            if not self.with_depth:
                self.left_paths.append(left)
                self.right_paths.append(right)
            else:
                # first check that the depth files exists
                left_depth = self._get_corresponding_depth_filename(left, self.depth_type)
                right_depth = self._get_corresponding_depth_filename(right, self.depth_type)
                if left_depth is not None and right_depth is not None and os.path.exists(left_depth):
                    self.left_paths.append(left)
                    self.right_paths.append(right)

        assert len(self.right_paths) == len(self.left_paths)
        # check mode
        if self.append_left_and_right_images:
            # append right images to left images
            self.left_paths += self.right_paths
            self.right_paths = []
        logging.info('Loaded {} files.'.format(len(self.left_paths)))

    def _get_corresponding_depth_filename(self, image_file, depth_type):
        if LEFT_STEREO_IMAGE_FOLDER in image_file:
            # left camera image
            depth_string = image_file.replace(LEFT_STEREO_IMAGE_FOLDER,
                                              'proj_depth/{type}/image_02'.format(type=depth_type))

            if depth_type not in PNG_DATASET:
                depth_string = depth_string.replace('png', 'npz')

            return depth_string

        elif RIGHT_STEREO_IMAGE_FOLDER in image_file:
            # right camera image
            depth_string = image_file.replace(RIGHT_STEREO_IMAGE_FOLDER,
                                              'proj_depth/{type}/image_03'.format(type=depth_type))

            if depth_type not in PNG_DATASET:
                depth_string = depth_string.replace('png', 'npz')

            return depth_string

        raise Exception(
            'Unknown image filename. Could not find left camera folder ({}) or right camera folder ({}) in image file {}'.format(
                LEFT_STEREO_IMAGE_FOLDER, RIGHT_STEREO_IMAGE_FOLDER, image_file))

    @staticmethod
    def _read_depth_as_np(depth_file, depth_type='velodyne'):
        """Reads Velodyne depth image

        Parameters
        ----------
        inputs: depth file name

        Returns
        -------
        outputs: np.float32
            Reads the ground truth depth image and converts it into a float32 Numpy array.
        """
        depth = np.load(depth_file)[depth_type + '_depth'].astype(np.float32)
        # Make sure we follow the Pytorch convention for Numpy images -> (H,W,C)
        depth = np.expand_dims(depth, axis=2)
        return depth

    @staticmethod
    def _read_groundtruth_depth_as_np(depth_file):
        """Reads ground truth depth image

        Parameters
        ----------
        inputs: depth file names

        Returns
        -------
        outputs: np.float32
            Reads the ground truth depth image and converts it into a float32 Numpy array.
        """
        depth_png = np.array(Image.open(depth_file), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth_png) > 255)
        depth = depth_png.astype(np.float) / 256.
        depth[depth_png == 0] = -1.
        # Make sure we follow the Pytorch convention for Numpy images -> (H,W,C)
        depth = np.expand_dims(depth, axis=2)
        return depth

    @staticmethod
    def _read_raw_calib_file(parent_folder):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        filepath = os.path.join(parent_folder, CAMERA_CALIBRATION_FILE)
        assert os.path.exists(filepath)
        data = read_calib_file(filepath)
        return data

    @staticmethod
    def _get_kitti_parent_folder_from_stereo_image(image_file):
        assert (LEFT_STEREO_IMAGE_FOLDER in image_file or RIGHT_STEREO_IMAGE_FOLDER in image_file)

        # the format is path_to_dataset/date_folder/drive_folder/image_02/data/*.png
        # we want path_to_dataset/date_folder
        # the camera calibration data is path_to_dataset/date_folder/CAMERA_CALIBRATION_FILE

        return os.path.abspath(os.path.join(image_file, "../../../.."))

    @staticmethod
    def _get_intrinsics(image_file, calib_data):

        """

        :param image_file: the KITTI dataset stereo image file for which to load the instrinsics
        :param calib_data: dictionary containing all the relevant calibration data for the run
        corresponding to the image_file
        :return: np array containing the camera intrinsics of size (3,3) and format:
                fx  0  cx
                0  fy  cy
                0   0   1
        """

        assert (LEFT_STEREO_IMAGE_FOLDER in image_file or RIGHT_STEREO_IMAGE_FOLDER in image_file)

        try:
            if LEFT_STEREO_IMAGE_FOLDER in image_file:
                p_rect = np.reshape(calib_data['P_rect_02'], (3, 4))
                return p_rect[:, :3]
            elif RIGHT_STEREO_IMAGE_FOLDER in image_file:
                p_rect = np.reshape(calib_data['P_rect_03'], (3, 4))
                return p_rect[:, :3]
        except KeyError as e:
            print('Exception {}\n File{}\n Calib Data {}\n\n'.format(e, image_file, calib_data))

    def _read_depth(self, sample_name):
        if self.depth_type in  ['velodyne', 'packnet']:
            depth = KittiLoader._read_depth_as_np(sample_name, depth_type=self.depth_type)
        elif self.depth_type == 'groundtruth':
            depth = KittiLoader._read_groundtruth_depth_as_np(sample_name)
        return depth

    def load_image(self, file_path):
        # assuming this file exists
        return Image.open(file_path)

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = self.load_image(self.left_paths[idx])
        sample = {'idx': idx, 'left_image': left_image}
        parent_folder = KittiLoader._get_kitti_parent_folder_from_stereo_image(self.left_paths[idx])
        if parent_folder in self.calibration_cache:
            c_data = self.calibration_cache[parent_folder]
        else:
            c_data = KittiLoader._read_raw_calib_file(parent_folder)
            self.calibration_cache[parent_folder] = c_data
        sample['left_intrinsics'] = KittiLoader._get_intrinsics(self.left_paths[idx], c_data)
        sample['left_intrinsics_inv'] = np.linalg.inv(sample['left_intrinsics']).astype(np.float32)
        sample['left_fx'] = sample['left_intrinsics'][0, 0]
        sample['left_fy'] = sample['left_intrinsics'][1, 1]
        
        if self.with_depth:
            left_depth = self._read_depth(
                self._get_corresponding_depth_filename(self.left_paths[idx], self.depth_type))
            sample.update({'left_depth': left_depth})

        if self.data_transform:
            sample = self.data_transform(sample)
        return sample
