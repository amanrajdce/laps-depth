#!/bin/bash
export PYTHONPATH="$(pwd)"

name="weather_internet"
img_path="/media/ehdd_2t/amanraj/data/weather_internet"

#ckpt_path="$local_dir/train_hp_search-5kt8_mono2_15aug_X_newprob/RayModel_0_2020-05-26_07-57-501pjwg1f9/checkpoint_itr158000/model.ckpt-158000"
ckpt_path="/ceph/amanraj/results/golden_ckpts/"

python pba/test_depth_simple.py \
  --ckpt_path "$ckpt_path" \
  --image_path "$img_path" \
  --test_batch_size 1 \
  --scale_normalize \
  --name "$name" \
  --save_pred


# CUDA_VISIBLE_DEVICES=1 bash ./scripts/test_depth_simple.sh