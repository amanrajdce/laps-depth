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
"""Transforms used in the Augmentation Policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import random
import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image


PARAMETER_MAX = 10  # What is the max 'level' a transform could be predicted


def pil_wrap(img_data):
    """Convert the list of `img` numpy tensor to a PIL Image."""
    pil_img_data = []
    for img in img_data:
        pil_img_data.append(Image.fromarray(np.uint8(img * 255.0)).convert('RGBA'))

    return pil_img_data


def pil_unwrap(img_data, image_size):
    """Converts the PIL img to a numpy array."""
    pil_img_data = []
    for pil_img in img_data:
        pic_array = (np.array(pil_img.getdata()).reshape((image_size[0], image_size[1], 4)) / 255.0)
        # find locations where pixel is completely transparent and set it's RGB value to 0
        i1, i2 = np.where(pic_array[:, :, 3] == 0)
        pic_array[i1, i2] = [0, 0, 0, 0]
        pil_img_data.append(pic_array[:, :, :3])  # ignoring alpha channel now

    return pil_img_data


def apply_policy(policy, data, image_size):
    """Apply the `policy` to the numpy `img`.

  Args:
    policy: A list of tuples with the form (name, probability, level) where
      `name` is the name of the augmentation operation to apply, `probability`
      is the probability of applying the operation and `level` is what strength
      the operation to apply.
    data: Numpy image list that will have `policy` applied to it.
    image_size: (height, width) of image.

  Returns:
    The result of applying `policy` to `img`.
  """
    img_data = data[:-1]
    intrinsic = data[-1]
    pil_img_data = pil_wrap(img_data)
    for xform in policy:
        assert len(xform) == 3
        name, probability, level = xform
        xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(probability, level, image_size)
        pil_img_data = xform_fn(pil_img_data)

    pil_img_data = pil_unwrap(pil_img_data, image_size)

    return [pil_img_data, intrinsic]


def random_flip(x):
    """Flip the input x horizontally with 50% probability."""
    if np.random.rand(1)[0] > 0.5:
        return np.fliplr(x)
    return x


def zero_pad_and_crop(img, amount=4):
    """Zero pad by `amount` zero pixels on each side then take a random crop.

  Args:
    img: numpy image that will be zero padded and cropped.
    amount: amount of zeros to pad `img` with horizontally and verically.

  Returns:
    The cropped zero padded img. The returned numpy array will be of the same
    shape as `img`.
  """
    padded_img = np.zeros((img.shape[0] + amount * 2,
                           img.shape[1] + amount * 2, img.shape[2]))
    padded_img[amount:img.shape[0] + amount, amount:img.shape[1] + amount, :] = img
    top = np.random.randint(low=0, high=2 * amount)
    left = np.random.randint(low=0, high=2 * amount)
    new_img = padded_img[top:top + img.shape[0], left:left + img.shape[1], :]
    return new_img


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / PARAMETER_MAX


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


class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, pil_img_data):
        return self.f(pil_img_data)


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def pil_transformer(self, probability, level, image_size):
        def return_function(img_data):
            if random.random() < probability:
                if 'image_size' in inspect.getargspec(self.xform).args:
                    img_data = [self.xform(im, level, image_size) for im in img_data]
                else:
                    img_data = [self.xform(im, level) for im in img_data]

            return img_data

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)


################## Transform Functions ##################
identity = TransformT('identity', lambda pil_img, level: pil_img)
flip_lr = TransformT(
    'FlipLR', lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))

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


def _posterize_impl(pil_img, level):
    """Applies PIL Posterize to `pil_img`."""
    level = int_parameter(level, 4)
    return ImageOps.posterize(pil_img.convert('RGB'), 4 - level).convert('RGBA')


posterize = TransformT('Posterize', _posterize_impl)


def _crop_impl(pil_img, level, image_size, interpolation=Image.BILINEAR):
    """Applies a crop to `pil_img` with the size depending on the `level`."""
    cropped = pil_img.crop((level, level, image_size - level, image_size - level))
    resized = cropped.resize((image_size, image_size), interpolation)
    return resized


crop_bilinear = TransformT('CropBilinear', _crop_impl)


def _solarize_impl(pil_img, level):
    """Applies PIL Solarize to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had Solarize applied to it.
  """
    level = int_parameter(level, 256)
    return ImageOps.solarize(pil_img.convert('RGB'), 256 - level).convert('RGBA')


solarize = TransformT('Solarize', _solarize_impl)


def _enhancer_impl(enhancer):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""

    def impl(pil_img, level):
        v = float_parameter(level, 1.8) + .1  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl


color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))

ALL_TRANSFORMS = [
    auto_contrast, equalize, invert, posterize, solarize,
    color, contrast, brightness, sharpness, blur, smooth
]
# TODO crop_bilinear and flip_lr

NAME_TO_TRANSFORM = {t.name: t for t in ALL_TRANSFORMS}
TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()
