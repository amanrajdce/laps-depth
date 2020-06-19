#!/bin/bash
# shellcheck disable=SC2155
export PYTHONPATH="$(pwd)"
export COMET_DISABLE_AUTO_LOGGING=1

search_kitti() {
  local_dir="/ceph/amanraj/results/"
  data_root="$1"

  kitti_root="$data_root/kitti_processed"
  kitti_raw="$data_root/kitti_raw_eigen_test"
  train_file_path="$kitti_root/train_lite5000.txt"
  test_file_path="$kitti_raw/test_files_eigen.txt"
  gt_path="$data_root/kitti_eigen_gt/gt_depth.npy"

  name="search_train_5k_t2_mono2_newprob"

  python pba/search.py \
    --local_dir "$local_dir" \
    --kitti_root "$kitti_root" \
    --kitti_raw "$kitti_raw" \
    --train_file_path "$train_file_path" \
    --test_file_path "$test_file_path" \
    --gt_path "$gt_path" \
    --batch_size 8 --lr 0.0002 --lr_decay step \
    --checkpoint_freq 0 \
    --gpu 1 --cpu 3 --epochs 35 --num_samples 2 \
    --perturbation_interval 1 --log_iter 250 \
    --scale_normalize --name "$name" --monodepth2 #--use_style_aug

  # SIGNet was trained for approx 35 epochs.
  # batch_size=4, lr=0.0002, no lr_decay
  # CUDA_VISIBLE_DEVICES=2,3 bash ./scripts/search.sh
}

mode=${1:-local} # defaults to local mode of deployment

# shellcheck disable=SC2198
if [ "$mode" = "pod" ]; then
  search_kitti /mnt/data
elif [ "$mode" = "local" ]; then
  search_kitti /ceph/amanraj/data
else
  echo "error executing script!"
fi