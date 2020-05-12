#!/bin/bash
export PYTHONPATH="$(pwd)"
export COMET_DISABLE_AUTO_LOGGING=1

local_dir="/ceph/amanraj/results/"
data_root="/ceph/amanraj/data"

kitti_root="$data_root/kitti_processed"
kitti_raw="$data_root/kitti_raw_eigen_test"
test_file_path="$kitti_raw/test_files_eigen.txt"
gt_path="$data_root/kitti_eigen_gt/gt_depth.npy"

train_file_path="$kitti_root/train.txt"
#hp_policy="$local_dir/search_train_lite_10000/pbt_policy_00001.txt"

name="test"

python pba/train.py \
  --local_dir "$local_dir" \
  --kitti_root "$kitti_root" \
  --kitti_raw "$kitti_raw" \
  --train_file_path "$train_file_path" \
  --test_file_path "$test_file_path" \
  --gt_path "$gt_path" \
  --batch_size 8 --lr 0.0002 \
  --checkpoint_freq 1 --gpu 1 --cpu 2 --epochs 3 \
  --scale_normalize --log_iter 20 \
  --use_kitti_aug --name "$name" --monodepth2 --load_all
  #--use_hp_policy --hp_policy "$hp_policy" --hp_policy_epochs 30 --disable_comet

#--lr_decay step
# SIGNet was trained for approx 25 epochs.
# batch_size=8, lr=0.0002, no lr_decay

# CUDA_VISIBLE_DEVICES=0 bash ./scripts/test_train.sh