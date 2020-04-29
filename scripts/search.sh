#!/bin/bash
export PYTHONPATH="$(pwd)"

local_dir="/media/ehdd_2t/amanraj/results/"
data_root="/media/ehdd_2t/amanraj/data/struct2depth"

kitti_root="$data_root/kitti_processed"
kitti_raw="$data_root/kitti_raw"
train_file_path="$kitti_root/train_lite.txt"
test_file_path="$kitti_raw/test_files_eigen.txt"
gt_path="$data_root/kitti_eigen_gt/gt_depth.npy"

name="search_train_lite_no_grad_clipping"

python pba/search.py \
  --local_dir "$local_dir" \
  --kitti_root "$kitti_root" \
  --kitti_raw "$kitti_raw" \
  --train_file_path "$train_file_path" \
  --test_file_path "$test_file_path" \
  --gt_path "$gt_path" \
  --batch_size 8 --lr 0.0002 \
  --checkpoint_freq 0 \
  --gpu 1 --cpu 3 --epochs 30 --num_samples 2 --grad_clipping 0.0 \
  --perturbation_interval 3 \
  --enable_batch_norm --scale_normalize \
  --name "$name"

# SIGNet was trained for approx 35 epochs.
# batch_size=4, lr=0.0002, no lr_decay
