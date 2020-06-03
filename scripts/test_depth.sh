#!/bin/bash
export PYTHONPATH="$(pwd)"

kitti_test_name="kitti_raw_eigen_test"
#kitti_test_name="kitti_raw_eigen_test_weather"
#kitti_test_name="kitti_raw_eigen_test_weather_single"

local_dir="/ceph/amanraj/results"
data_root="/ceph/amanraj/data"
kitti_raw="$data_root/$kitti_test_name"
test_file_path="$kitti_raw/test_files_eigen.txt"
gt_path="$data_root/kitti_eigen_gt/gt_depth.npy"

ckpt_path="$local_dir/train_hp_search-5kt8_mono2_15aug_X_newprob/RayModel_0_2020-05-26_07-57-501pjwg1f9/checkpoint_itr158000/model.ckpt-158000"

python pba/test_depth.py \
  --ckpt_path "$ckpt_path" \
  --kitti_raw "$kitti_raw" \
  --test_file_path "$test_file_path" \
  --gt_path "$gt_path" \
  --test_batch_size 1 \
  --scale_normalize \
  --name "$kitti_test_name" \
  --save_pred


# CUDA_VISIBLE_DEVICES=1 bash ./scripts/test_depth.sh