"""Root package info"""

__version__ = '0.1.0'
__author__ = 'TRI ML team'
__license__ = 'MIT'
__copyright__ = 'Copyright (c) 2019-2020, TRI'
__homepage__ = 'https://github.com/TRI-ML/packnet-sfm'
__docs__ = 'packnet-sfm is a library for monocular depth and pose estimation'
__long_docs__ = """
Official [PyTorch](https://pytorch.org/) implementation of _self-supervised_
monocular depth estimation methods invented by the ML Team at
[Toyota Research Institute (TRI)](https://www.tri.global/),
in particular for _PackNet_:
[**3D Packing for Self-Supervised Monocular Depth Estimation (CVPR 2020 oral)**](https://arxiv.org/abs/1905.02693),
*Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos and Adrien Gaidon*.

Although self-supervised (i.e. trained only on monocular videos),
PackNet outperforms other self, semi, and fully supervised methods.
Furthermore, it gets better with input resolution and number of parameters, generalizes better, and can run in real-time (with TensorRT). See [References](#references) for more info on our models.

"""
