[<img src="/media/figs/tri-logo.png" width="30%">](https://www.tri.global/)

This repository contains code for the following papers:

## 3D Packing for Self-Supervised Monocular Depth Estimation
*Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos and Adrien Gaidon*

[**[Full paper]**](https://arxiv.org/abs/1905.02693) 
[**[YouTube]**](https://www.youtube.com/watch?v=b62iDkLgGSI)

## Robust Semi-Supervised Monocular Depth Estimation with Reprojected Distances
*Vitor Guizilini, Jie Li, Rares Ambrus, Sudeep Pillai and Adrien Gaidon*

[**[Full paper]**](https://arxiv.org/abs/1910.01765) 
[**[YouTube]**](https://www.youtube.com/watch?v=cSwuF-XA4sg)

## Two Stream Networks for Self-Supervised Ego-Motion Estimation
*Rares Ambrus, Vitor Guizilini, Jie Li, Sudeep Pillai and Adrien Gaidon*

[**[Full paper]**](https://arxiv.org/abs/1910.01764) 

## Semantically-Guided Representation Learning for Self-Supervised Monocular Depth
*Vitor Guizilini, Rui Hou, Jie Li, Rares Ambrus and Adrien Gaidon*

[**[Full paper]**](https://arxiv.org/abs/2002.12319) 

## SuperDepth: Self-Supervised, Super-Resolved Monocular Depth Estimation
*Sudeep Pillai, Rares Ambrus and Adrien Gaidon*

[**[Full paper]**](https://arxiv.org/abs/1810.01849)
[**[YouTube]**](https://www.youtube.com/watch?v=jKNgBeBMx0I&t=33s)

## Contributions

- **PackNet**: A new convolutional network architecture for high-resolution self-supervised monocular depth estimation.  We propose new packing and unpacking blocks that jointly leverage 3D convolutions to learn representations that maximally propagate dense appearance and geometric information while still being able to run in real time.   

- **Weak Velocity Supervision**: A novel optional loss that can leverage the camera’s velocity when available (e.g. from cars, robots, mobile phones) to solve the inherent scale ambiguity in monocular vision.    

- **Dense Depth for Automated Driving (DDAD)**: A new dataset that leverages diverse logs from a fleet of well-calibrated self-driving cars equipped with cameras and high-accuracy long-range LiDARs.  Compared toexisting benchmarks, DDAD enables much more accurate depth evaluation at range, which is key for high resolution monocular depth estimation methods.
 
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

## How to Use
 
### Step 1: Clone this repository

```
git clone https://github.com/vguizilini/packnet-sfm.git
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
- Self-Supervised (192x640, K)
    ```
    wget https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet_MR_selfsup_K.pth.tar
    ```
- Self-Supervised (192x640, CS)
    ```
    wget https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet_MR_selfsup_CS.pth.tar
    ```
- Self-Supervised Scale-Aware (192x640, CS &rightarrow; K)
    ```
    wget https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet_MR_velsup_CStoK.pth.tar
    ```
- Semi-Supervised (Annotated depth maps) (192x640, CS &rightarrow; K)
    ```
    wget https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet_MR_semisup_CStoK.pth.tar
    ```

### Step 5: Inference
```
bash evaluate_kitti.sh
```

### License

The source code is released under the [MIT license](LICENSE.md).

### Citations
Depending on the application, please use the following citations when referencing our work:

```
@misc{packnet-sfm-selfsup,
  author = {Vitor Guizilini and Rares Ambrus and Sudeep Pillai and Allan Raventos and Adrien Gaidon},
  title = {3D Packing for Self-Supervised Monocular Depth Estimation},
  archivePrefix = {arXiv:1905.02693},
  primaryClass = {cs.CV}
  year = {2019},
}
```

```
@proceedings{packnet-sfm-semisup,
  author = {Vitor Guizilini and Jie Li and Rares Ambrus and Sudeep Pillai and Adrien Gaidon},
  title = {Robust Semi-Supervised Monocular Depth Estimation with Reprojected Distances},
  booktitle = {In Proceedings of the 3rd Annual Conference on Robot Learning (CoRL)}
  month = {October},
  year = {2019},
}
```

```
@proeedings{packnet-sfm-twostream,
  author = {Rares Ambrus and Vitor Guizilini and Jie Li and Sudeep Pillai and Adrien Gaidon},
  title = {{Two Stream Networks for Self-Supervised Ego-Motion Estimation}},
  booktitle = {In Proceedings of the 3rd Annual Conference on Robot Learning (CoRL)}
  month = {October},
  year = {2019},
}
```

```
@proceedings{packnet-sfm-semguided,
  author = {Vitor Guizilini and Rui Hou and Jie Li and Rares Ambrus and Adrien Gaidon},
  title = {Semantically-Guided Representation Learning for Self-Supervised Monocular Depth},
  booktitle = {In Proceedings of the 8th International Conference on Learning Representations (ICLR)}
  month = {April},
  year = {2020},
}
```

```
@proceedings{superdepth,
  author = {Sudeep Pillai and Rares Ambrus and Adrien Gaidon},
  title = {SuperDepth: Self-Supervised, Super-Resolved Monocular Depth Estimation},
  booktitle = {In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)}
  month = {May},
  year = {2019},
}
```
