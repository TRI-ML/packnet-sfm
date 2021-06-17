# Copyright 2021 Toyota Research Institute.  All rights reserved.


import MinkowskiEngine as ME
import torch


def sparsify_features(x):
    """
    Sparsify features

    Parameters
    ----------
    x : Dense feature map [B,C,H,W]

    Returns
    -------
    Sparse feature map (features only in valid coordinates)
    """
    b, c, h, w = x.shape

    u = torch.arange(w).reshape(1, w).repeat([h, 1])
    v = torch.arange(h).reshape(h, 1).repeat([1, w])
    uv = torch.stack([v, u], 2).reshape(-1, 2)

    coords = [uv] * b
    feats = [feats.permute(1, 2, 0).reshape(-1, c) for feats in x]
    coords, feats = ME.utils.sparse_collate(coords=coords, feats=feats)
    return ME.SparseTensor(coordinates=coords, features=feats, device=x.device)


def sparsify_depth(x):
    """
    Sparsify depth map

    Parameters
    ----------
    x : Dense depth map [B,1,H,W]

    Returns
    -------
    Sparse depth map (range values only in valid pixels)
    """
    b, c, h, w = x.shape

    u = torch.arange(w, device=x.device).reshape(1, w).repeat([h, 1])
    v = torch.arange(h, device=x.device).reshape(h, 1).repeat([1, w])
    uv = torch.stack([v, u], 2)

    idxs = [(d > 0)[0] for d in x]

    coords = [uv[idx] for idx in idxs]
    feats = [feats.permute(1, 2, 0)[idx] for idx, feats in zip(idxs, x)]
    coords, feats = ME.utils.sparse_collate(coords=coords, feats=feats)
    return ME.SparseTensor(coordinates=coords, features=feats, device=x.device)


def densify_features(x, shape):
    """
    Densify features from a sparse tensor

    Parameters
    ----------
    x : Sparse tensor
    shape : Dense shape [B,C,H,W]

    Returns
    -------
    Dense tensor containing sparse information
    """
    stride = x.tensor_stride
    coords, feats = x.C.long(), x.F
    shape = (shape[0], shape[2] // stride[0], shape[3] // stride[1], feats.shape[1])
    dense = torch.zeros(shape, device=x.device)
    dense[coords[:, 0],
          coords[:, 1] // stride[0],
          coords[:, 2] // stride[1]] = feats
    return dense.permute(0, 3, 1, 2).contiguous()


def densify_add_features_unc(x, s, u, shape):
    """
    Densify and add features considering uncertainty

    Parameters
    ----------
    x : Dense tensor [B,C,H,W]
    s : Sparse tensor
    u : Sparse tensor with uncertainty
    shape : Dense tensor shape

    Returns
    -------
    Densified sparse tensor with added uncertainty
    """
    stride = s.tensor_stride
    coords, feats = s.C.long(), s.F
    shape = (shape[0], shape[2] // stride[0], shape[3] // stride[1], feats.shape[1])

    dense = torch.zeros(shape, device=s.device)
    dense[coords[:, -1],
          coords[:, 0] // stride[0],
          coords[:, 1] // stride[1]] = feats
    dense = dense.permute(0, 3, 1, 2).contiguous()

    mult = torch.ones(shape, device=s.device)
    mult[coords[:, -1],
         coords[:, 0] // stride[0],
         coords[:, 1] // stride[1]] = 1.0 - u.F
    mult = mult.permute(0, 3, 1, 2).contiguous()

    return x * mult + dense


def map_add_features(x, s):
    """
    Map dense features to sparse tensor and add them.

    Parameters
    ----------
    x : Dense tensor [B,C,H,W]
    s : Sparse tensor

    Returns
    -------
    Sparse tensor with added dense information in valid areas
    """
    stride = s.tensor_stride
    coords = s.coords.long()
    feats = x.permute(0, 2, 3, 1)
    feats = feats[coords[:, -1],
                  coords[:, 0] // stride[0],
                  coords[:, 1] // stride[1]]
    return ME.SparseTensor(coords=coords, feats=feats + s.feats,
                           coords_manager=s.coords_man, force_creation=True,
                           tensor_stride=s.tensor_stride)
