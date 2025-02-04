#!/bin/bash

# shellcheck disable=SC2155
export PYTHONPATH="$(pwd)"
export COMET_DISABLE_AUTO_LOGGING=1

train_hp_kitti() {
  local_dir="/ceph/amanraj/results"
  data_root="$1"

  kitti_root="$data_root/kitti_processed"
  kitti_raw="$data_root/kitti_raw_eigen_test"
  train_file_path="$kitti_root/train.txt"
  test_file_path="$kitti_raw/test_files_eigen.txt"
  gt_path="$data_root/kitti_eigen_gt/gt_depth.npy"

  name="train_hp_random_policy_schedule"
  hp_policy="$local_dir/search_train_5k_t8_mono2_15aug_X_newprob/pbt_policy_$2.txt"
  # restore="$local_dir/train_hp_search-5kt2-max-25q-mono2_higher/checkpoint_17/model.ckpt-17"

  python pba/train.py \
    --local_dir "$local_dir" \
    --kitti_root "$kitti_root" \
    --kitti_raw "$kitti_raw" \
    --train_file_path "$train_file_path" \
    --test_file_path "$test_file_path" \
    --gt_path "$gt_path" \
    --name "$name" --scale_normalize \
    --checkpoint_freq 1 --checkpoint_iter 2000 --checkpoint_iter_after 10 \
    --batch_size 8 --lr 0.0002 --lr_decay step \
    --gpu 1 --cpu 4 --epochs 35 --log_iter 1000 --monodepth2 \
    --use_hp_policy --hp_policy "$hp_policy" --hp_policy_epochs 35 --flatten
    # --restore $restore \
    # --disable_comet
    # CUDA_VISIBLE_DEVICES=1 bash pod/train_mono2_rand_policy.sh local 00002
}

mode=${1:-local} # defaults to local mode of deployment
policy=${2:-00000}  # defaults to pbt_policy_00000.txt

# shellcheck disable=SC2198
if [ "$mode" = "pod" ]; then
  train_hp_kitti /mnt/data "$policy"
elif [ "$mode" = "local" ]; then
  train_hp_kitti /ceph/amanraj/data "$policy"
else
  echo "error executing script!"
fi