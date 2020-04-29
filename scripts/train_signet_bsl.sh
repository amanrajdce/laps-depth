#!/bin/bash
export PYTHONPATH="$(pwd)"
export COMET_DISABLE_AUTO_LOGGING=1

local_dir="/media/ehdd_2t/amanraj/results/"
data_root="/media/ehdd_2t/amanraj/data/struct2depth"

kitti_root="$data_root/kitti_processed"
kitti_raw="$data_root/kitti_raw"
train_file_path="$kitti_root/train_test.txt"
test_file_path="$kitti_raw/test_files_eigen.txt"
gt_path="$data_root/kitti_eigen_gt/gt_depth.npy"

#name="train_full_signet_bsl"
name="test"

python pba/train.py \
  --local_dir "$local_dir" \
  --kitti_root "$kitti_root" \
  --kitti_raw "$kitti_raw" \
  --train_file_path "$train_file_path" \
  --test_file_path "$test_file_path" \
  --gt_path "$gt_path" \
  --batch_size 8 --lr 0.0002 \
  --checkpoint_freq 1 --gpu 1 --cpu 3 --epochs 10 \
  --scale_normalize --log_iter 100 \
  --no_aug_policy --name "$name"

#--use_kitti_aug
# SIGNet was trained for approx 35 epochs.
# batch_size=4, lr=0.0002, no lr_decay