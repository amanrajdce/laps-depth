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
import torch
import tensorflow as tf
from torchvision.transforms import ToTensor, ToPILImage

import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image  # pylint:disable=g-multiple-import

from pba.augmentation_transforms import random_flip, zero_pad_and_crop  # pylint: disable=unused-import
from pba.augmentation_transforms import TransformFunction
from pba.augmentation_transforms import pil_wrap, pil_unwrap  # pylint: disable=unused-import
from pba.augmentation_transforms import PARAMETER_MAX, float_parameter, int_parameter  # pylint: disable=unused-import
from pba.augmentation_transforms import _posterize_impl, _solarize_impl, _enhancer_impl
from pba.augmentationsX import add_rain, add_snow, add_fog, add_speed


# PyTorch Tensor <-> PIL Image transforms:
toTensor = ToTensor()
toPIL = ToPILImage()


def apply_policy(policy, data, image_size, style_augmentor=None, fliplr=False, cutout=False, verbose=False):
    """
    Apply the `policy` to the numpy `img`.

    Args:
        policy: A list of tuples with the form (name, probability, level) where
        `name` is the name of the augmentation operation to apply, `probability`
        is the probability of applying the operation and `level` is what strength
        the operation to apply.
        data: Numpy image list that will have `policy` applied to it.
        image_size: (height, width) of image.
        count: number of operations to apply
        style_augmentor: function that augments data with randomized style
        fliplr: randomly flip the images
        cutout: randomly apply a random cutout
        verbose: Whether to print applied augmentations.

    Returns:
        The result of applying `policy` to `data`.
    """
    # PBA cifar10 policy modified
    # count = np.random.choice([0, 1, 2, 3], p=[0.10, 0.20, 0.30, 0.40]) # old
    count = np.random.choice([0, 1, 2, 3], p=[0.20, 0.20, 0.50, 0.10])  # new
    #count = np.random.choice([0, 1, 2], p=[0.30, 0.50, 0.20])  # new2
    # count = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.5, 0.0])  # original

    if count != 0:
        data['img_data'] = pil_wrap(data['img_data'])
        policy = copy.copy(policy)
        random.shuffle(policy)
        for xform in policy:
            assert len(xform) == 3
            name, probability, level = xform
            assert 0. <= probability <= 1.
            assert 0 <= level <= PARAMETER_MAX
            xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(probability, level, image_size, style_augmentor)
            data, res = xform_fn(data)
            if verbose and res:
                tf.logging.info("Op: {}, Magnitude: {}, Prob: {}".format(name, level, probability))
            count -= res
            assert count >= 0
            if count == 0:
                break
        data['img_data'] = pil_unwrap(data['img_data'], image_size)  # apply pil_unwrap on imgs only

    # should be used only when flipr is not being learned in augmentation policy
    if fliplr and random.random() > 0.5:
        data['img_data'] = [np.fliplr(img) for img in data['img_data']]

    # should be used only when cutout is not being learned in augmentation policy
    if cutout and random.random() > 0.5:
        data['img_data'] = [cutout_numpy(img, size=20) for img in data['img_data']]

    return data


def cutout_numpy(img, size=16):
    """Apply cutout with mask of shape `size` x `size` to `img`.

  The cutout operation is from the paper https://arxiv.org/abs/1708.04552.
  This operation applies a `size`x`size` mask of zeros to a random location
  within `img`.

  Args:
    img: Numpy image that cutout will be applied to.
    size: Height/width of the cutout mask that will be

  Returns:
    A numpy tensor that is the result of applying the cutout mask to `img`.
  """
    img_height, img_width, num_channels = (img.shape[0], img.shape[1], img.shape[2])
    assert len(img.shape) == 3
    mask, _, _ = create_cutout_mask(img_height, img_width, num_channels, size)
    return img * mask


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def pil_transformer(self, probability, level, image_size, style_augmentor):
        """Builds augmentation function which returns resulting image and whether augmentation was applied."""

        def return_function(data):
            res = False
            if random.random() < probability:
                func_args = inspect.getargspec(self.xform).args
                if 'data' in func_args and 'style_augmentor' in func_args:
                    data = self.xform(data, level, style_augmentor)  # for _style_aug_impl
                elif 'data' in func_args and 'image_size' in func_args:
                    data = self.xform(data, level, image_size)  # for _scale_crop_pil_impl
                elif 'image_size' in func_args:
                    data['img_data'] = [self.xform(im, level, image_size) for im in data['img_data']]
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


def _random_style_aug_impl(data, level, style_augmentor):
    img_data = data['img_data']
    alpha = float_parameter(level, maxval=0.50)
    # style function, toTensor->converts values to 0 to 1
    im_torch = torch.cat([toTensor(im.convert('RGB')).unsqueeze(0) for im in img_data], dim=0)

    # choose a random style:
    im_restyled = style_augmentor(im_torch, alpha=alpha)
    im_restyled = im_restyled.squeeze().cpu()  # value range is 0 to 1

    im_restyled = [toPIL(im_restyled[idx, :, :, :]) for idx in range(im_restyled.size(0))]
    data['img_data'] = [im.convert('RGBA') for im in im_restyled]  # values range is 0 to 255

    return data


def _rain_impl(data, level, image_size):
    img_data = data['img_data']
    # rainy days are usually shady
    bright_coeff = 0.8
    img_data = [ImageEnhance.Brightness(img.convert('RGB')).enhance(bright_coeff) for img in img_data]
    img_data = [np.array(img) for img in img_data]
    slant = int_parameter(level, maxval=20)
    if random.random() < 0.5:
        slant *= -1

    img_data = add_rain(img_data, slant=slant)
    img_data = [toPIL(img).convert('RGBA') for img in img_data]
    data['img_data'] = img_data

    return data


def _snow_impl(data, level, image_size):
    img_data = data['img_data']
    img_data = [np.array(img.convert('RGB')) for img in img_data]
    snow_coeff = float_parameter(level, maxval=0.5)
    img_data = add_snow(img_data, snow_coeff=snow_coeff)
    img_data = [toPIL(img).convert('RGBA') for img in img_data]
    data['img_data'] = img_data

    return data


def _fog_impl(data, level, image_size):
    img_data = data['img_data']
    img_data = [np.array(img.convert('RGB')) for img in img_data]
    fog_coeff = 0.3 + float_parameter(level, maxval=0.4)
    img_data = add_fog(img_data, fog_coeff=fog_coeff)
    img_data = [toPIL(img).convert('RGBA') for img in img_data]
    data['img_data'] = img_data

    return data


def _speed_blur_impl(data, level, image_size):
    img_data = data['img_data']
    img_data = [np.array(img.convert('RGB')) for img in img_data]
    img_data = add_speed(img_data, speed_coeff=0.1)
    img_data = [toPIL(img).convert('RGBA') for img in img_data]
    data['img_data'] = img_data

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
random_style = TransformT('RandomStyle', _random_style_aug_impl)
rain = TransformT('Rain', _rain_impl)
snow = TransformT('Snow', _snow_impl)
fog = TransformT('Fog', _fog_impl)
speed_blur = TransformT('SpeedBlur', _speed_blur_impl)

"""
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
    #random_style,
    blur,   # Added new stuffs from here
    smooth,
    edge_enhance,
    #contour,
    flip_lr,
    scale_crop,
    rain,  # augmentations X from here
    snow,
    fog,
    speed_blur,
]
# TODO crop_bilinear

"""
HP_TRANSFORMS = [
    brightness,
    color,
    sharpness,
    equalize,
    auto_contrast,
    smooth,
    edge_enhance,
    snow,
    fog,
    speed_blur,
]

NAME_TO_TRANSFORM = collections.OrderedDict((t.name, t) for t in HP_TRANSFORMS)
HP_TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()
NUM_HP_TRANSFORM = len(HP_TRANSFORM_NAMES)
