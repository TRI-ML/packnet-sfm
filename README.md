[<img src="/media/figs/tri-logo.png" width="25%">](https://www.tri.global/)

## PackNet: Self-Supervised Deep Network for Monocular Depth Estimation

- [How to Use](#how-to-use)
- [Qualitative Results](#qualitative-results)
- [License](#license)
- [References](#references)

Reference [PyTorch](https://pytorch.org/) implementations of _self-supervised_ monocular depth estimation methods invented by the ML Team at [Toyota Research Institute (TRI)](https://www.tri.global/), in particular for our paper titled: **3D Packing for Self-Supervised Monocular Depth Estimation (CVPR 2020 oral)**, 
*Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos and Adrien Gaidon*, [**[Full paper]**](https://arxiv.org/abs/1905.02693)

[![Alt text](https://img.youtube.com/vi/b62iDkLgGSI/0.jpg)](https://www.youtube.com/watch?v=b62iDkLgGSI)

This paper introduced **PackNet**: a new deep convolutional network architecture for high-resolution monocular depth estimation. PackNet relies on symmetric packing and unpacking blocks to jointly learn to compress and decompress detail-preserving representations using 3D convolutions. Although self-supervised (i.e. trained only on monocular videos), our method outperforms other self, semi, and fully supervised methods. Furthermore, it gets better with input resolution and number of parameters, generalizes better, and can run in real-time.

Our CVPR paper also introduced **Weak Velocity Supervision**: a novel optional loss that can leverage the camera’s velocity when available (e.g. from cars, robots, mobile phones) to solve the inherent scale ambiguity in monocular structure-from-motion (SfM).

Finally, we also release **Dense Depth for Automated Driving ([DDAD](https://github.com/TRI-ML/DDAD))**: A new dataset that leverages diverse logs from a fleet of well-calibrated self-driving cars equipped with cameras and high-accuracy long-range LiDARs.  Compared to existing benchmarks, DDAD enables much more accurate depth evaluation at range.

PackNet can be trained from scratch in a fully self-supervised way (from video only), in a semi-supervised way (with sparse lidar using our novel reprojected 3D loss), and it can also use a fixed pre-trained semantic segmentation network to guide the representation learning further.

Training code will be released soon. See also the full list of [References](#references) below for more details.

## How to Use
 
### Step 1: Clone this repository

```
git clone https://github.com/TRI-ML/packnet-sfm.git
```

### Step 2: Create symbolic link to data folder

```
sudo ln -s path/to/data /data
```

### Step 3: Download datasets into /data/datasets

#### [KITTI_raw](http://www.cvlibs.net/datasets/kitti/raw_data.php) 
- For convenience, we also provide the pre-computed depth maps used in our papers (unzip into the same root folder)
    ```
    wget https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/depth_maps/KITTI_raw_velodyne.tar.gz
    wget https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/depth_maps/KITTI_raw_groundtruth.tar.gz
    ```
    
### Step 4: Download pre-trained models into /data/models

#### KITTI
- Self-Supervised, 192x640, Kitti (K)
    ```
    wget https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet_MR_selfsup_K.pth.tar
    ```
- Self-Supervised, 192x640, CityScapes (CS)
    ```
    wget https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet_MR_selfsup_CS.pth.tar
    ```
- Self-Supervised Scale-Aware, 192x640, CS &rightarrow; K
    ```
    wget https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet_MR_velsup_CStoK.pth.tar
    ```
- Semi-Supervised (Annotated depth maps), 192x640, CS &rightarrow; K
    ```
    wget https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet_MR_semisup_CStoK.pth.tar
    ```

### Step 5: Inference
```
bash evaluate_kitti.sh
```

## Qualitative Results

### Self-Supervised - KITTI

<img src="/media/figs/teaser27.png" width="49%"> <img src="/media/figs/teaser51.png" width="49%">
<img src="/media/figs/teaser305.png" width="49%"> <img src="/media/figs/teaser291.png" width="49%">

### Self-Supervised - DDAD

<img src="/media/figs/ddad1.png" width="49%"> <img src="/media/figs/ddad2.png" width="49%">
<img src="/media/figs/ddad3.png" width="49%"> <img src="/media/figs/ddad4.png" width="49%">

### Semi-Supervised - KITTI

<img src="/media/figs/beams_full.jpg" width="32%" height="170cm"> <img src="/media/figs/beams_64.jpg" width="32%"  height="170cm"> <img src="/media/figs/beams_32.jpg" width="32%" height="170cm">
<img src="/media/figs/beams_16.jpg" width="32%"  height="170cm"> <img src="/media/figs/beams_8.jpg" width="32%"  height="170cm"> <img src="/media/figs/beams_4.jpg" width="32%"  height="170cm">

### Semantically-Guided Self-Supervised Depth - KITTI

<img src="/media/figs/semguided.png" width="98%">>

### Solving the Infinite Depth Problem

<img src="/media/figs/infinite_depth.png" width="98%">

## License

The source code is released under the [MIT license](LICENSE.md).

## References

Depending on the application, please use the following citations when referencing our work:

#### 3D Packing for Self-Supervised Monocular Depth Estimation (CVPR 2020 oral) 
*Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos and Adrien Gaidon*, [**[paper]**](https://arxiv.org/abs/1905.02693), [**[video]**](https://www.youtube.com/watch?v=b62iDkLgGSI)
```
@inproceedings{packnet,
  author = {Vitor Guizilini and Rares Ambrus and Sudeep Pillai and Allan Raventos and Adrien Gaidon},
  title = {3D Packing for Self-Supervised Monocular Depth Estimation},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  primaryClass = {cs.CV}
  year = {2020},
}
```

#### Semantically-Guided Representation Learning for Self-Supervised Monocular Depth (ICLR 2020)
*Vitor Guizilini, Rui Hou, Jie Li, Rares Ambrus and Adrien Gaidon*, [**[paper]**](https://arxiv.org/abs/2002.12319)
```
@inproceedings{packnet-semguided,
  author = {Vitor Guizilini and Rui Hou and Jie Li and Rares Ambrus and Adrien Gaidon},
  title = {Semantically-Guided Representation Learning for Self-Supervised Monocular Depth},
  booktitle = {International Conference on Learning Representations (ICLR)}
  month = {April},
  year = {2020},
}
```

#### Robust Semi-Supervised Monocular Depth Estimation with Reprojected Distances (CoRL 2019 spotlight)
*Vitor Guizilini, Jie Li, Rares Ambrus, Sudeep Pillai and Adrien Gaidon*, [**[paper]**](https://arxiv.org/abs/1910.01765),[**[video]**](https://www.youtube.com/watch?v=cSwuF-XA4sg)
```
@inproceedings{packnet-semisup,
  author = {Vitor Guizilini and Jie Li and Rares Ambrus and Sudeep Pillai and Adrien Gaidon},
  title = {Robust Semi-Supervised Monocular Depth Estimation with Reprojected Distances},
  booktitle = {Conference on Robot Learning (CoRL)}
  month = {October},
  year = {2019},
}
```

#### Two Stream Networks for Self-Supervised Ego-Motion Estimation (CoRL 2019 spotlight)
*Rares Ambrus, Vitor Guizilini, Jie Li, Sudeep Pillai and Adrien Gaidon*, [**[paper]**](https://arxiv.org/abs/1910.01764) 
```
@inproceedings{packnet-twostream,
  author = {Rares Ambrus and Vitor Guizilini and Jie Li and Sudeep Pillai and Adrien Gaidon},
  title = {{Two Stream Networks for Self-Supervised Ego-Motion Estimation}},
  booktitle = {Conference on Robot Learning (CoRL)}
  month = {October},
  year = {2019},
}
```

#### SuperDepth: Self-Supervised, Super-Resolved Monocular Depth Estimation (ICRA 2019)
*Sudeep Pillai, Rares Ambrus and Adrien Gaidon*, [**[paper]**](https://arxiv.org/abs/1810.01849), [**[video]**](https://www.youtube.com/watch?v=jKNgBeBMx0I&t=33s)
```
@inproceedings{superdepth,
  author = {Sudeep Pillai and Rares Ambrus and Adrien Gaidon},
  title = {SuperDepth: Self-Supervised, Super-Resolved Monocular Depth Estimation},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)}
  month = {May},
  year = {2019},
}
```
