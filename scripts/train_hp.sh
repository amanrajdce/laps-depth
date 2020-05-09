#!/bin/bash

export PYTHONPATH="$(pwd)"
export COMET_DISABLE_AUTO_LOGGING=1

local_dir="/ceph/amanraj/results"
data_root="/ceph/amanraj/data"

kitti_root="$data_root/kitti_processed"
kitti_raw="$data_root/kitti_raw"
train_file_path="$kitti_root/train.txt"
test_file_path="$kitti_raw/test_files_eigen.txt"
gt_path="$data_root/kitti_eigen_gt/gt_depth.npy"

name="train_hp_search-train-5k-t2-max-25q"
#hp_policy="$PWD/schedules/rcifar10_16_kitti.txt"
hp_policy="$local_dir/search_train_5k_t2_max_25q/pbt_policy_00000.txt"


python pba/train.py \
  --local_dir "$local_dir" \
  --kitti_root "$kitti_root" \
  --kitti_raw "$kitti_raw" \
  --train_file_path "$train_file_path" \
  --test_file_path "$test_file_path" \
  --gt_path "$gt_path" \
  --batch_size 8 --lr 0.0002 --lr_decay step \
  --checkpoint_freq 1 --gpu 1 --cpu 3 --epochs 35 \
  --name "$name" --scale_normalize --log_iter 1000 \
  --use_hp_policy --hp_policy "$hp_policy" --hp_policy_epochs 35 --disable_comet


# SIGNet was trained for approx 35 epochs.
# batch_size=4, lr=0.0002, no lr_decay
# CUDA_VISIBLE_DEVICES=1 bash ./scripts/train_hp.sh
