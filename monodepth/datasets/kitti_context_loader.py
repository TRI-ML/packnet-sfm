# Copyright 2020 Toyota Research Institute.  All rights reserved.

import glob
import os

import numpy as np

from monodepth.datasets.kitti import KittiLoader
from monodepth.geometry.utils import invert_pose_numpy


class KittiContextLoader(KittiLoader):
    """KITTI data loader which handles temporal context.`

      Parameters
      ----------
      root_dir          : dataset path
      file_list         : split file containing relative path to data
      mode              : whether to load stereo or monocular data (choices: 'stereo', 'mono')
      train             : bool
      data_transform    : transform to be applied on each data sample before returning it
      depth_type        : KITTI depth type. choices [None, 'groundtruth', 'velodyne']
      back_context      : Number of frames before the current frame to return.
                        E.g. if the current frame is T, and back_context=2, the loader will return
                        [T-2, T-1, T]
      forward_context   : Number of frames after the current frame to return.
      min_speed         : Min speed (m/s) for a frame to be returned. Speed is part of the KITTI
                        dataset and available for each frame.
      precomputed_speed : bool - whether to load the speed for each frame from a pre-computed file.
                        If False, the speed will be loaded from the KITTI dataset (slower than
                        reading from a precomputed file)
      with_pose         : bool - if True, the pose of each frame will be returned.
      strides           : list of strides, denoting the number of frames to skip for each sample.
                        E.g. for frame T, strides=[1,2], back_context=0, forward_context=2,
                        the dataloader will return the following samples:
                        [T,T+1,T+2] (for stride 1), [T,T+2,T+4] (for stride 2)

      """

    def __init__(self, root_dir, file_list, data_transform=None,
                 depth_type=None, append_left_and_right_images=False,
                 with_semantic=False, back_context=0, forward_context=0):
                 
        KittiLoader.__init__(self, root_dir, file_list, data_transform, depth_type,
                             append_left_and_right_images)
        self.back_context = back_context
        self.forward_context = forward_context
        self._cache = {}
        self.forward_context_paths = []
        self.back_context_paths = []
        self.with_context = (back_context != 0 or forward_context != 0)
        self.split = file_list.split('/')[-1].split('.')[0]
        assert back_context >= 0 and forward_context >= 0
        # Note: we are assuming a KITTI_semantic folder with the same structure
        # as KITTI_raw and in the same folder
        self.with_semantic = with_semantic

        if self.with_context:
            left_paths_with_context = []
            for i, f in enumerate(self.left_paths):
                back_context_idxs, forward_context_idxs = self._get_context_for_sample(f,
                                                                                        depth_type,
                                                                                        back_context,
                                                                                        forward_context)
                if back_context_idxs is not None and forward_context_idxs is not None:
                    left_paths_with_context.append(self.left_paths[i])
                    # reverse so we get [idx-back_context:idx]
                    self.back_context_paths.append(back_context_idxs[::-1])
                    self.forward_context_paths.append(forward_context_idxs)

            # update left and right paths
            self.left_paths = left_paths_with_context
            print('After context filtering {} samples are left.'.format(len(self.left_paths)))

    def _get_context_for_sample(self, sample_name, depth_type, back_context, forward_context,
                                stride=1):
        # note this assumes that both self.left_paths[idx] and self.rights_paths[idx] exist
        base, ext = os.path.splitext(os.path.basename(sample_name))
        parent_folder = os.path.dirname(sample_name)
        f_idx = int(base)
        # look for context in the range [f_idx-back_context:f_ids+forward_context]
        # for each sample, check depth as well
        # if no depth, continue

        # back context (store only for left image -> we can generate the other paths automatically)
        back_context_idxs = []
        forward_context_idxs = []

        # check num files in folder
        if parent_folder in self._cache:
            max_num_files = self._cache[parent_folder]
        else:
            max_num_files = len(glob.glob(os.path.join(parent_folder, '*' + ext)))
            self._cache[parent_folder] = max_num_files

        # check bounds
        if (f_idx - back_context * stride) < 0 or (
                f_idx + forward_context * stride) >= max_num_files:
            # context out of available bounds
            return None, None

        # back context
        c_idx = f_idx
        while len(back_context_idxs) < back_context and c_idx > 0:
            c_idx -= stride
            filename = KittiContextLoader._get_next_file(c_idx, sample_name)

            if os.path.exists(filename):
                back_context_idxs.append(c_idx)

        if c_idx < 0:
            return None, None

        # forward context
        c_idx = f_idx
        while len(forward_context_idxs) < forward_context and c_idx < max_num_files:
            c_idx += stride
            filename = KittiContextLoader._get_next_file(c_idx, sample_name)

            if os.path.exists(filename):
                forward_context_idxs.append(c_idx)

        if c_idx >= max_num_files:
            return None, None

        return back_context_idxs, forward_context_idxs

    @staticmethod
    def _get_next_file(idx, filename):
        base, ext = os.path.splitext(os.path.basename(filename))
        return os.path.join(os.path.dirname(filename), str(idx).zfill(len(base)) + ext)

    def _get_context_files_for_sample_name_and_idxs(self, sample_name, idxs):
        image_context_paths = [KittiContextLoader._get_next_file(i, sample_name) for i in idxs]
        if self.with_depth:
            depth_context_paths = [
                self._get_corresponding_depth_filename(f, self.depth_type) for f in
                image_context_paths]
            return image_context_paths, depth_context_paths
        else:
            return image_context_paths, None

    def __getitem__(self, idx):
        left_image = self.load_image(self.left_paths[idx])
        sample = {'left_image': left_image, 'idx': idx}

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

        if self.with_semantic:
            # Note: we are assuming a KITTI_semantic folder with the same structure
            # as KITTI_raw and in the same folder
            sample['left_semantic'] = self.load_image(self.left_paths[idx].replace('_raw', '_semantic'))
            sample['left_semantic'] = np.array(sample['left_semantic'])

        if self.with_context:
            all_context_idxs = self.back_context_paths[idx] + self.forward_context_paths[idx]
            image_context_paths_left, depth_context_paths_left = \
                self._get_context_files_for_sample_name_and_idxs(
                    self.left_paths[idx], all_context_idxs)

            image_context_left = [self.load_image(f) for f in image_context_paths_left]
            sample.update({'left_image_context': image_context_left})

        sample['filename'] = '%s_%010d' % (self.split, idx),

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample