#!/bin/bash
export PYTHONPATH="$(pwd)"

local_dir="/ceph/amanraj/results"
data_root="/ceph/amanraj/data"
kitti_raw="$data_root/kitti_raw_eigen_test"
test_file_path="$kitti_raw/test_files_eigen.txt"
gt_path="$data_root/kitti_eigen_gt/gt_depth.npy"
name="kitti_evaluation"

ckpt_path="$local_dir/paper/train_hp_search-5kt2-max-25q-mono2/policy1_run1/checkpoint_16/model.ckpt-16"

python pba/test_depth.py \
  --ckpt_path "$ckpt_path" \
  --kitti_raw "$kitti_raw" \
  --test_file_path "$test_file_path" \
  --gt_path "$gt_path" \
  --test_batch_size 1 \
  --scale_normalize \
  --name "$name" \
  --save_pred


# CUDA_VISIBLE_DEVICES=1 bash ./scripts/test_depth.sh