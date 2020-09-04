#! /usr/bin/env python2

"""
I/O script to save and load the data coming with the MPI-Sintel low-level
computer vision benchmark.

For more details about the benchmark, please visit www.mpi-sintel.de

CHANGELOG:
v1.0 (2015/02/03): First release

Copyright (c) 2015 Jonas Wulff
Max Planck Institute for Intelligent Systems, Tuebingen, Germany

"""

# Requirements: Numpy as PIL/Pillow
import numpy as np
from PIL import Image

# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

def flow_read(filename):
    """ Read optical flow from file, return (U,V) tuple. 
    
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' flow_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    tmp = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width*2))
    u = tmp[:,np.arange(width)*2]
    v = tmp[:,np.arange(width)*2 + 1]
    return u,v

def flow_write(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

   
def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def depth_write(filename, depth):
    """ Write depth to file. """
    height,width = depth.shape[:2]
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    
    depth.astype(np.float32).tofile(f)
    f.close()


def disparity_write(filename,disparity,bitdepth=16):
    """ Write disparity to file.

    bitdepth can be either 16 (default) or 32.

    The maximum disparity is 1024, since the image width in Sintel
    is 1024.
    """
    d = disparity.copy()

    # Clip disparity.
    d[d>1024] = 1024
    d[d<0] = 0

    d_r = (d / 4.0).astype('uint8')
    d_g = ((d * (2.0**6)) % 256).astype('uint8')

    out = np.zeros((d.shape[0],d.shape[1],3),dtype='uint8')
    out[:,:,0] = d_r
    out[:,:,1] = d_g

    if bitdepth > 16:
        d_b = (d * (2**14) % 256).astype('uint8')
        out[:,:,2] = d_b

    Image.fromarray(out,'RGB').save(filename,'PNG')


def disparity_read(filename):
    """ Return disparity read from filename. """
    f_in = np.array(Image.open(filename))
    d_r = f_in[:,:,0].astype('float64')
    d_g = f_in[:,:,1].astype('float64')
    d_b = f_in[:,:,2].astype('float64')

    depth = d_r * 4 + d_g / (2**6) + d_b / (2**14)
    return depth


#def cam_read(filename):
#    """ Read camera data, return (M,N) tuple.
#    
#    M is the intrinsic matrix, N is the extrinsic matrix, so that
#
#    x = M*N*X,
#    with x being a point in homogeneous image pixel coordinates, X being a
#    point in homogeneous world coordinates.
#    """
#    txtdata = np.loadtxt(filename)
#    intrinsic = txtdata[0,:9].reshape((3,3))
#    extrinsic = textdata[1,:12].reshape((3,4))
#    return intrinsic,extrinsic
#
#
#def cam_write(filename,M,N):
#    """ Write intrinsic matrix M and extrinsic matrix N to file. """
#    Z = np.zeros((2,12))
#    Z[0,:9] = M.ravel()
#    Z[1,:12] = N.ravel()
#    np.savetxt(filename,Z)

def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

def cam_write(filename, M, N):
    """ Write intrinsic matrix M and extrinsic matrix N to file. """
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    M.astype('float64').tofile(f)
    N.astype('float64').tofile(f)
    f.close()


def segmentation_write(filename,segmentation):
    """ Write segmentation to file. """

    segmentation_ = segmentation.astype('int32')
    seg_r = np.floor(segmentation_ / (256**2)).astype('uint8')
    seg_g = np.floor((segmentation_ % (256**2)) / 256).astype('uint8')
    seg_b = np.floor(segmentation_ % 256).astype('uint8')

    out = np.zeros((segmentation.shape[0],segmentation.shape[1],3),dtype='uint8')
    out[:,:,0] = seg_r
    out[:,:,1] = seg_g
    out[:,:,2] = seg_b

    Image.fromarray(out,'RGB').save(filename,'PNG')


def segmentation_read(filename):
    """ Return disparity read from filename. """
    f_in = np.array(Image.open(filename))
    seg_r = f_in[:,:,0].astype('int32')
    seg_g = f_in[:,:,1].astype('int32')
    seg_b = f_in[:,:,2].astype('int32')

    segmentation = (seg_r * 256 + seg_g) * 256 + seg_b
    return segmentation


