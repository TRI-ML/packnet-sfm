# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import numpy as np

from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from monodepth.logging import printcolor

# File template for loading
IMAGE_REGEX = '{:09d}'


def get_idx(filename):
    """Get the index of the image filename (specific to a session) per-line.
    For e.g. berlin/508/image_02/data/berlin_000508_000012_leftImg8bit.png
    index corresponds to 12.

    Parameters
    ----------
    filename: str
        Image filename

    Returns
    ----------
    index: int
        Index of the filename
    """
    return int(os.path.splitext(filename)[0])


def change_idx(idx, filename):
    """Get the image filename for the index provided.

    Parameters
    ----------
    idx: int
        Image index
    filename: str
        Image filename

    Returns
    ----------
    filename: str
        Filename of the new image with the corresponding index
    """
    _, ext = os.path.splitext(os.path.basename(filename))
    return IMAGE_REGEX.format(idx) + ext


def read_files(directory, ext=['.png', '.jpg', '.jpeg'], skip_empty=True):
    """Read files recursively within a directory and return the directory
    structure as a dictionary.

    Parameters
    ----------
    directory: str
        Root directory
    ext: list
        List of acceptable file extensions.
    skip_empty: bool, (default=True)
        Only create dictionary key/value if the directory is empty.

    Returns
    ----------
    files: dict
        Directory structure as a dict
    """
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


def read_file_list(filename, ext=['.png', '.jpg', '.jpeg'], skip_empty=True):
    """Read files from the file-list and return the directory
    structure as a dictionary.

    Parameters
    ----------
    filename: str
        File list name
    ext: list
        List of acceptable file extensions.
    skip_empty: bool, (default=True)
        Only create dictionary key/value if the directory is empty.

    Returns
    ----------
    files: dict
        Directory structure as a dict
    """
    files = defaultdict(list)
    for entry in open(filename, 'r').read().splitlines():
        dirname, basename = os.path.split(entry)
        files[dirname].append(basename)
    for k in files:
        if not len(files[k]):
            files.pop(k)
    return files


class ImageSequenceLoader(Dataset):
    def __init__(self, root_dir, file_list=None, data_transform=None,
                 forward_context=0, backward_context=0,
                 strides=[1], dataset_idx=None):
        """Image sequence data loader which handles temporal context.
        Supported image formats are .png, .jpg, .jpeg

        The dataset directory structure should be as follows:
        <root_dir>/<unique_session_name>/<%09d.png>

        For example:
        >> root_dir/
           root_dir/session1/intrinsics.json
           root_dir/session1/000000001.png
           root_dir/session1/000000002.png
           root_dir/session1/...
           root_dir/session2/intrinsics.json
           root_dir/session2/000000001.png
           root_dir/session2/000000002.png
           root_dir/...
           root_dir/sessionN/intrinsics.json
           root_dir/sessionN/000000001.png
           root_dir/sessionN/000000002.png
           root_dir/sessionN/...

        Parameters
        ----------
        root_dir: str
            Dataset path
        file_list: str
            Split file containing relative path to data
        data_transform: Data Transform
            Transform to be applied on each data sample before returning it
        forward_context: int, (default=0)
            Number of frames after the current frame to return.
        back_context: int, (default=0)
            Number of frames before the current frame to return.
            E.g. if the current frame is T, and back_context=2, the loader will return
            [T-2, T-1, T]
        strides: list, (default=[1])
            List of strides, denoting the number of frames to skip for each sample.
            (currently not supported)
        dataset_idx: int, (default=None)
            Identify dataset index loader for mixed batch training

        Notes
        ----------
            1. This loader assumes that consecutive frame indices (t-1,t,t+1)
            are present in the same session.
            2. The loader does not check for file existence when file_list is
            provided.
        """
        super().__init__()
        assert len(strides) == 1 and strides[0] == 1
        assert isinstance(strides, list)
        self.dataset_idx = dataset_idx
        self.root_dir = root_dir
        self.forward_context = forward_context
        self.backward_context = backward_context
        self.stride = 1

        # Support training from image sequence directory, or via file splits
        file_tree = read_files(root_dir)
        if file_list is not None:
            self.tree = read_file_list(file_list)
        else:
            self.tree = file_tree
        self.sessions = self.tree.keys()

        self.calib = {}
        self.files = []
        for (k,v) in self.tree.items():
            self.calib[k] = self._get_calibration(k)
            plen = len(v)
            v = sorted(v)
            file_set = set(file_tree[k])
            files = [fname for fname in v if self._has_context(fname, file_set)]
            self.tree[k] = files
            self.files.extend([(k, fname) for fname in files])

        printcolor('ImageSequence: {}'.format(self.root_dir))
        printcolor('\tSessions: {}'.format(len(self.sessions)))
        printcolor('\tDataset size: {}'.format(len(self.files)))
        printcolor('\tImages size: {}'.format(sum([len(v) for v in self.tree.values()])))
        self.data_transform = data_transform

    def __len__(self):
        return len(self.files)

    def _has_context(self, filename, file_set):
        """Check if the filename (fname) has context files in the file_set

        Parameters
        ----------
        filename: str
            Filename
        file_set: set
            Set of files that exist

        Returns
        ----------
        bool
            Return whether the filename contains the context files list, after context check.
        """
        context_paths = self._get_context_file_paths(filename)
        return all([f in file_set for f in context_paths])

    def _get_calibration(self, session):
        """Get the calibration file for the video session.

        Parameters
        ----------
        session: str
            Video session

        Returns
        ----------
        calib: dict
            Calibration dict with 'K' and 'Kinv' keys
        """
        filename = os.path.join(self.root_dir, session, 'intrinsics.json')
        if os.path.exists(filename):
            raise NotImplementedError()

        # Load calibration by using image size
        filename = os.path.join(self.root_dir, session, self.tree[session][0])
        im = Image.open(filename)
        W, H = im.size
        K = np.array([[1000., 0, W / 2 - 0.5],
                      [0, 1000., H / 2 - 0.5],
                      [0, 0, 1]])
        Kinv = K.copy()
        Kinv[0, 0] = 1. / K[0, 0]
        Kinv[1, 1] = 1. / K[1, 1]
        Kinv[0, 2] = -K[0, 2] / K[0, 0]
        Kinv[1, 2] = -K[1, 2] / K[1, 1]
        return {'K': K, 'Kinv': Kinv}

    def _get_context_file_paths(self, filename):
        """ Return RGB context files given a filename

        Parameters
        ----------
        filename: str
            Filename

        Returns
        ----------
        filenames: list
            Context image filenames
        """
        fidx = get_idx(filename)
        idxs = list(np.arange(-self.backward_context * self.stride, 0, self.stride)) + \
               list(np.arange(0, self.forward_context * self.stride, self.stride) + self.stride)
        return [change_idx(fidx + i, filename) for i in idxs]

    def _read_rgb_context_files(self, session, filename):
        """Read context images for the given index.

        Parameters
        ----------
        session: str
            Session name
        filename: str
            Filename

        Returns
        ----------
        filenames: list
            List of context RGB images
        """
        context_paths = self._get_context_file_paths(filename)
        return [Image.open(os.path.join(self.root_dir, session, filename)) for filename in context_paths]

    def _read_rgb_file(self, session, filename):
        """Return RGB image given an index

        Parameters
        ----------
        session: str
            Session name
        filename: str
            Filename

        Returns
        ----------
        image: PIL.Image
            RGB image
        """
        return Image.open(os.path.join(self.root_dir, session, filename))

    def __getitem__(self, idx):
        """Return RGB image given an index

        Parameters
        ----------
        idx: int
            Image index

        Returns
        ----------
        sample: dict
            RGB image sample along with corresponding intrinsics, index
        """
        session, filename = self.files[idx]
        calib = self.calib[session]
        K, Kinv = calib['K'], calib['Kinv']
        sample = {'left_intrinsics': K,
                  'left_intrinsics_inv': Kinv,
                  'left_fx': K[0, 0],
                  'left_fy': K[1, 1],
                  'baseline': 0,
                  'idx': idx
                  }
        if self.dataset_idx is not None:
            sample.update({'dataset_idx' : self.dataset_idx})

        sample['left_image'] = self._read_rgb_file(session, filename)
        sample['left_image_context'] = self._read_rgb_context_files(session, filename)

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample
