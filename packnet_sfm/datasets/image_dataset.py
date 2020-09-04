
import re
from collections import defaultdict
import os

from torch.utils.data import Dataset
import numpy as np
from packnet_sfm.utils.image import load_image

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def dummy_calibration(image):
    w, h = [float(d) for d in image.size]
    return np.array([[1000. , 0.    , w / 2. - 0.5],
                     [0.    , 1000. , h / 2. - 0.5],
                     [0.    , 0.    , 1.          ]])

def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(load_image(file), dtype=int)
    assert (np.max(depth_png) > 255), 'Wrong .png depth file'
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return np.expand_dims(depth, axis=2)

def get_idx(filename):
    return int(re.search(r'\d+', filename).group())

def read_files(directory, ext=('.png', '.jpg', '.jpeg'), skip_empty=True):
    files = defaultdict(list)
    for entry in os.scandir(directory):
        relpath = os.path.relpath(entry.path, directory)
        if entry.is_dir():
            d_files = read_files(entry.path, ext=ext, skip_empty=skip_empty)
            if skip_empty and not len(d_files):
                continue
            files[relpath] = d_files[entry.path]
        elif entry.is_file():
            if ext is None or entry.path.lower().endswith(tuple(ext)):
                files[directory].append(relpath)
    return files

########################################################################################################################
#### DATASET
########################################################################################################################

class ImageDataset(Dataset):
    def __init__(self, root_dir, file_list,  train=True, data_transform=None,
                 forward_context=0, back_context=0, strides=(1,),
                 depth_type='groundtruth', **kwargs):
        super().__init__()
        # Asserts

        self.root_dir = root_dir
        self.split = split

        self.backward_context = back_context
        self.forward_context = forward_context
        self.has_context = self.backward_context + self.forward_context > 0
        self.strides = 1

        self.train = True
        self.with_depth = depth_type is not '' and depth_type is not None

        self.files = []

        with open(file_list, "r") as f:
            data = f.readlines()

        self.paths = []
        # Get file list from data
        for i, fname in enumerate(data):
            path = os.path.join(root_dir, fname.split()[0])
            if not self.with_depth:
                self.paths.append(path)
            else:
                # Check if the depth file exists
                depth = self._get_depth_file(path)
                if depth is not None and os.path.exists(depth):
                    self.paths.append(path)

        self.data_transform = data_transform

    def __len__(self):
        return len(self.paths)

    def _change_idx(self, idx, filename):
        _, ext = os.path.splitext(os.path.basename(filename))
        return self.split.format(idx) + ext

    def _has_context(self, filename, file_set):
        context_paths = self._get_context_file_paths(filename)
        return all([f in file_set for f in context_paths])

    def _get_context_file_paths(self, filename):
        fidx = get_idx(filename)
        idxs = list(np.arange(-self.backward_context * self.strides, 0, self.strides)) + \
               list(np.arange(0, self.forward_context * self.strides, self.strides) + self.strides)
        return [self._change_idx(fidx + i, filename) for i in idxs]

    def _read_rgb_context_files(self, session, filename):
        context_paths = self._get_context_file_paths(filename)
        return [load_image(os.path.join(self.root_dir, session, filename))
                for filename in context_paths]

    def _read_rgb_file(self, session, filename):
        return load_image(os.path.join(self.root_dir, session, filename))

    def __getitem__(self, idx):
        session, filename = self.paths[idx]
        image = self._read_rgb_file(session, filename)

        sample = {
            'idx': idx,
            'filename': '%s_%s' % (session, os.path.splitext(filename)[0]),
            #
            'rgb': image,
            'intrinsics': dummy_calibration(image)
        }

        if self.has_context:
            sample['rgb_context'] = \
                self._read_rgb_context_files(session, filename)

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample

########################################################################################################################
