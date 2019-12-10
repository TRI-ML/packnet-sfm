# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Example of evaluation script for KITTI

make docker-evaluate-depth \
MODEL=/data/models/packnet/PackNet_MR_selfsup_K.pth.tar \
INPUT_PATH=/data/datasets/KITTI_raw/data_splits/eigen_test_files.txt \
DEPTH_TYPE=velodyne \
CROP=garg \
SAVE_OUTPUT=output


