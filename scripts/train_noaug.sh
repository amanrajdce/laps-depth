#!/bin/bash
export PYTHONPATH="$(pwd)"

local_dir="$PWD/results/"
data_root="/media/ehdd_2t/amanraj/data/struct2depth"

kitti_root="$data_root/kitti_processed"
kitti_raw="$data_root/kitti_raw"
train_file_path="$kitti_root/train.txt"
test_file_path="$kitti_raw/test_files_eigen.txt"
gt_path="$data_root/kitti_eigen_gt/gt_depth.npy"

name="train_full_noaug" # training on lite without any augmentation

python pba/train.py \
  --local_dir "$local_dir" \
  --kitti_root "$kitti_root" \
  --kitti_raw "$kitti_raw" \
  --train_file_path "$train_file_path" \
  --test_file_path "$test_file_path" \
  --gt_path "$gt_path" \
  --batch_size 8 --lr 0.0002 \
  --checkpoint_freq 1 --gpu 1 --cpu 3 --epochs 30 \
  --enable_batch_norm --scale_normalize \
  --no_aug_policy --name "$name"

# SIGNet was trained for approx 35 epochs.
# batch_size=4, lr=0.0002, no lr_decay