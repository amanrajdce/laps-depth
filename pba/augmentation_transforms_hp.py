# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transforms used in the PBA Augmentation Policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import collections
import inspect
import random
import tensorflow as tf

import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image  # pylint:disable=g-multiple-import

from pba.augmentation_transforms import random_flip, zero_pad_and_crop  # pylint: disable=unused-import
from pba.augmentation_transforms import TransformFunction
from pba.augmentation_transforms import pil_wrap, pil_unwrap  # pylint: disable=unused-import
from pba.augmentation_transforms import PARAMETER_MAX, float_parameter  # pylint: disable=unused-import
from pba.augmentation_transforms import _posterize_impl, _solarize_impl, _enhancer_impl


def apply_policy(policy, data, image_size, verbose=False):
    """
    Apply the `policy` to the numpy `img`.

    Args:
        policy: A list of tuples with the form (name, probability, level) where
        `name` is the name of the augmentation operation to apply, `probability`
        is the probability of applying the operation and `level` is what strength
        the operation to apply.
        data: Numpy image list that will have `policy` applied to it.
        image_size: (height, width) of image.
        verbose: Whether to print applied augmentations.

    Returns:
        The result of applying `policy` to `data`.
    """
    # PBA cifar10 policy modified
    count = np.random.choice([0, 1, 2, 3], p=[0.10, 0.20, 0.30, 0.40])
    # count = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.5, 0.0])

    if count != 0:
        data['img_data'] = pil_wrap(data['img_data'])
        policy = copy.copy(policy)
        random.shuffle(policy)
        for xform in policy:
            assert len(xform) == 3
            name, probability, level = xform
            assert 0. <= probability <= 1.
            assert 0 <= level <= PARAMETER_MAX
            xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(probability, level, image_size)
            data, res = xform_fn(data)
            if verbose and res:
                tf.logging.info("Op: {}, Magnitude: {}, Prob: {}".format(name, level, probability))
            count -= res
            assert count >= 0
            if count == 0:
                break
        data['img_data'] = pil_unwrap(data['img_data'], image_size)  # apply pil_unwrap on imgs only
        return data
    else:
        return data


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def pil_transformer(self, probability, level, image_size):
        """Builds augmentation function which returns resulting image and whether augmentation was applied."""

        def return_function(data):
            res = False
            if random.random() < probability:
                func_args = inspect.getargspec(self.xform).args
                if 'data' in func_args and 'image_size' in func_args:
                    data = self.xform(data, level, image_size)  # transform that modifies intrinsic also
                elif 'image_size' in func_args:
                    data['img_data'] = [self.xform(im, level, image_size) for im in data['img_data']]  # transform that modifies only imgs
                else:
                    data['img_data'] = [self.xform(im, level) for im in data['img_data']]
                res = True
            return data, res

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)

    def str(self):
        return self.name


################## Transform Functions ##################
identity = TransformT('identity', lambda pil_img, level: pil_img)

# pylint:disable=g-long-lambda
auto_contrast = TransformT(
    'AutoContrast', lambda pil_img, level: ImageOps.autocontrast(pil_img.convert('RGB')).convert('RGBA')
)
equalize = TransformT(
    'Equalize', lambda pil_img, level: ImageOps.equalize(pil_img.convert('RGB')).convert('RGBA')
)
invert = TransformT(
    'Invert', lambda pil_img, level: ImageOps.invert(pil_img.convert('RGB')).convert('RGBA')
)
# pylint:enable=g-long-lambda
blur = TransformT('Blur', lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))
smooth = TransformT('Smooth', lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))
edge_enhance = TransformT('EdgeEnhance', lambda pil_img, level: pil_img.filter(ImageFilter.EDGE_ENHANCE))
contour = TransformT('Contour', lambda pil_img, level: pil_img.filter(ImageFilter.CONTOUR))
posterize = TransformT('Posterize', _posterize_impl)


def _crop_impl(pil_img, level, image_size, interpolation=Image.BILINEAR):
    """Applies a crop to `pil_img` with the size depending on the `level`."""
    cropped = pil_img.crop((level, level, image_size - level, image_size - level))
    resized = cropped.resize((image_size, image_size), interpolation)
    return resized


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / PARAMETER_MAX)


def create_cutout_mask(img_height, img_width, num_channels, size):
    """Creates a zero mask used for cutout of shape `img_height` x `img_width`.

  Args:
    img_height: Height of image where cutout mask will be applied to.
    img_width: Width of image where cutout mask will be applied to.
    num_channels: Number of channels in the image.
    size: size of the zeros mask.

  Returns:
    A mask of shape `img_height` x `img_width` with all ones except for a
    square of zeros of shape `size` x `size`. This mask is meant to be
    elementwise multiplied with the original image. Additionally returns
    the `upper_coord` and `lower_coord` which specify where the cutout mask
    will be applied.
  """
    # Sample center where cutout mask will be applied
    height_loc = np.random.randint(low=10, high=img_height-10)
    width_loc = np.random.randint(low=10, high=img_width-10)

    # Determine upper left and lower right corners of patch
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (min(img_height, height_loc + size // 2), min(img_width, width_loc + size // 2))

    mask_height = lower_coord[0] - upper_coord[0]
    mask_width = lower_coord[1] - upper_coord[1]
    assert mask_height > 0
    assert mask_width > 0
    assert 0 <= upper_coord[0] < img_height
    assert 0 <= upper_coord[1] < img_width
    assert 0 <= lower_coord[0] < img_height
    assert 0 <= lower_coord[1] < img_width

    mask = np.ones((img_height, img_width, num_channels))
    zeros = np.zeros((mask_height, mask_width, num_channels))
    mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = zeros
    return mask, upper_coord, lower_coord


def _cutout_pil_impl(pil_img, level, image_size):
    """Apply cutout to pil_img at the specified level."""
    size = int_parameter(level, 20)
    if size <= 0:
        return pil_img
    img_height, img_width, num_channels = (image_size[0], image_size[1], 3)
    _, upper_coord, lower_coord = create_cutout_mask(img_height, img_width, num_channels, size)

    # tf.logging.info("img_height: {}, img_width:{}".format(img_height, img_width))
    # tf.logging.info("upper: {}, lower: {}".format(upper_coord, lower_coord))
    # tf.logging.info("pil_height: {}, pil_width:{}".format(pil_img.height, pil_img.width))

    pixels = pil_img.load()  # create the pixel map
    for i in range(upper_coord[0], lower_coord[0]):  # for every row:
        for j in range(upper_coord[1], lower_coord[1]):  # for every col:
            pixels[j, i] = (125, 122, 113, 0)  # set the colour accordingly
    return pil_img


def _scale_crop_pil_impl(data, level, image_size):
    """Apply scaling and random crop"""
    h, w = image_size
    scale = 1.05 + float_parameter(level, maxval=0.15)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # scaling
    data['img_data'] = [pil_img.resize((new_w, new_h), resample=2) for pil_img in data['img_data']]  # resample=2 for bilinear
    data['intrinsic'][0, 0] *= scale  # fx
    data['intrinsic'][1, 1] *= scale  # fy
    data['intrinsic'][0, 2] *= scale  # cx
    data['intrinsic'][1, 2] *= scale  # cy

    # random crop
    offset_y = np.random.randint(0, new_h - h + 1)
    offset_x = np.random.randint(0, new_w - w + 1)
    data['img_data'] = [pil_img.crop((offset_x, offset_y, offset_x + w, offset_y + h)) for pil_img in data['img_data']]
    data['intrinsic'][0, 2] -= float(offset_x)  # cx
    data['intrinsic'][1, 2] -= float(offset_y)  # cy

    return data


cutout = TransformT('Cutout', _cutout_pil_impl)
crop_bilinear = TransformT('CropBilinear', _crop_impl)
solarize = TransformT('Solarize', _solarize_impl)
color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))
scale_crop = TransformT('ScaleCrop', _scale_crop_pil_impl)
flip_lr = TransformT('FlipLR', lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))

HP_TRANSFORMS = [
    brightness,
    color,
    invert,
    sharpness,
    posterize,
    solarize,
    equalize,
    auto_contrast,
    cutout,
    contrast,
    #blur,   # Added new stuffs from here
    #smooth,
    #edge_enhance,
    #contour,
    #flip_lr,
    #scale_crop
]
# TODO crop_bilinear

NAME_TO_TRANSFORM = collections.OrderedDict((t.name, t) for t in HP_TRANSFORMS)
HP_TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()
NUM_HP_TRANSFORM = len(HP_TRANSFORM_NAMES)
