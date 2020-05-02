#!/bin/bash
export PYTHONPATH="$(pwd)"
export COMET_DISABLE_AUTO_LOGGING=1

local_dir="/ceph/amanraj/results/"
data_root="/ceph/amanraj/data"

kitti_root="$data_root/kitti_processed"
kitti_raw="$data_root/kitti_raw"
train_file_path="$kitti_root/train.txt"
test_file_path="$kitti_raw/test_files_eigen.txt"
gt_path="$data_root/kitti_eigen_gt/gt_depth.npy"

name="train_full_autoaugment_run2"

python pba/train.py \
  --local_dir "$local_dir" \
  --kitti_root "$kitti_root" \
  --kitti_raw "$kitti_raw" \
  --train_file_path "$train_file_path" \
  --test_file_path "$test_file_path" \
  --gt_path "$gt_path" \
  --batch_size 8 --lr 0.0002 --grad_clipping 0.0 \
  --checkpoint_freq 1 --gpu 1 --cpu 3 --epochs 30 \
  --enable_batch_norm --scale_normalize \
  --policy_dataset cifar10 \
  --name "$name"

# SIGNet was trained for approx 35 epochs.
# batch_size=4, lr=0.0002, no lr_decay
