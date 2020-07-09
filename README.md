# Learning Augmentation Policy Schedules for Unsuperivsed Depth Estimation

## Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
pip install ray
pip install numpy==1.16.1
pip install comet_ml
pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install requests
```
We ran our experiments with PyTorch-gpu 1.5, CUDA 9.1, Python 3.5, TensorFlow-gpu 1.11, Ray 0.8.4, Cudnn7 and Ubuntu 18.04.

You can pull the docker image with the above setup from : ```amanraj42/tensorflow-gpu-ray:v1.11``` . We also provide
a docker file in this repository.

## Checkpoints 
Our checkpoints are available at : https://drive.google.com/drive/u/3/folders/19QAZam6TdoRyTdCjbdoripW4xldp4ZV2
Method names are in Table 4.2 and Table 4.3 in the thesis.

## Prediction for a single image
Download the checkpoints and use the one interested with: 

```shell
python test_depth_simple.py --scale_normalize --image_path path/to/image.jpg --ckpt_path path/to/model.ckpt-*
```

## Search : Learning Data Augmentation Policy Schedules

## Training : Full scale training on KITTI
