# Copyright 2020 Toyota Research Institute.  All rights reserved.

set -e
set -x

MODEL_PATH=$1  # /path/to/model
INPUT_PATH=$2  # /path/to/input
DEPTH_TYPE=$3  # depth type used for inference (e.g. 'velodyne' or 'groundtruth' for KITTI)
CROP=$4        # crop to be used (e.g. 'garg' for KITTI)
SAVE_OUTPUT=$5 # /path/to/save_folder

python scripts/evaluate_depth.py \
       --pretrained_model $MODEL_PATH \
       --input_path $INPUT_PATH \
       --depth_type $DEPTH_TYPE \
       --crop $CROP \
       --save_output $SAVE_OUTPUT
