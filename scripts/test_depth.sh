#!/bin/bash
export PYTHONPATH="$(pwd)"

local_dir="/ceph/amanraj/results"
data_root="/ceph/amanraj/data"
kitti_raw="$data_root/kitti_raw"
test_file_path="$kitti_raw/test_files_eigen.txt"
gt_path="$data_root/kitti_eigen_gt/gt_depth.npy"

ckpt_dir="$local_dir/train_full_noaug/RayModel_0_2020-04-20_03-54-42ywb7axqx"

name="kitti_evaluation"

python pba/test_depth.py \
  --ckpt_dir "$ckpt_dir" \
  --kitti_raw "$kitti_raw" \
  --test_file_path "$test_file_path" \
  --gt_path "$gt_path" \
  --test_batch_size 1 \
  --scale_normalize \
  --name "$name"


# CUDA_VISIBLE_DEVICES=0 bash ./scripts/test_depth.sh