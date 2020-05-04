#!/bin/bash
export PYTHONPATH="$(pwd)"
export COMET_DISABLE_AUTO_LOGGING=1

local_dir="/ceph/amanraj/results/"
data_root="/ceph/amanraj/data"

kitti_root="$data_root/kitti_processed"
kitti_raw="$data_root/kitti_raw"
test_file_path="$kitti_raw/test_files_eigen.txt"
gt_path="$data_root/kitti_eigen_gt/gt_depth.npy"

train_file_path="$kitti_root/train_test.txt"

name="test"

python pba/train.py \
  --local_dir "$local_dir" \
  --kitti_root "$kitti_root" \
  --kitti_raw "$kitti_raw" \
  --train_file_path "$train_file_path" \
  --test_file_path "$test_file_path" \
  --gt_path "$gt_path" \
  --batch_size 8 --lr 0.0002 --num_workers 12 \
  --checkpoint_freq 1 --gpu 1 --cpu 4 --epochs 1 \
  --scale_normalize --log_iter 20 \
  --no_aug_policy --use_kitti_aug --name "$name"

#--lr_decay step
# SIGNet was trained for approx 25 epochs.
# batch_size=8, lr=0.0002, no lr_decay

#CUDA_VISIBLE_DEVICES=3 bash ./scripts/test_train.sh
