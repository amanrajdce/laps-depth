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
"""AutoAugment augmentation policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Unique operations: Invert, Contrast, AutoContrast, Equalize, Solarize
def good_policies_svhn():
    return [[('Equalize', 0.6, 7), ('Invert', 0.2, 3)],
            [('Equalize', 0.6, 7), ('Invert', 0.7, 5)],
            [('Equalize', 0.6, 5), ('Solarize', 0.6, 6)],
            [('Invert', 0.9, 3), ('Equalize', 0.6, 3)],
            [('Equalize', 0.6, 1), ('Equalize', 0.6, 7)],
            [('Equalize', 0.6, 7), ('AutoContrast', 0.8, 3)],
            [('Equalize', 0.6, 7), ('Invert', 0.4, 5)],
            [('Equalize', 0.6, 7), ('Solarize', 0.2, 6)],
            [('Invert', 0.9, 6), ('AutoContrast', 0.8, 1)],
            [('Equalize', 0.6, 3), ('Equalize', 0.6, 7)],
            [('Equalize', 0.6, 7), ('Solarize', 0.3, 3)],
            [('Equalize', 0.6, 7), ('Invert', 0.7, 4)],
            [('Equalize', 0.9, 5), ('Equalize', 0.6, 7)],
            [('Invert', 0.9, 4), ('Equalize', 0.6, 7)],
            [('Contrast', 0.3, 3), ('Equalize', 0.6, 7)],
            [('Invert', 0.8, 5), ('Equalize', 0.6, 7)],
            [('Equalize', 0.6, 7), ('Solarize', 0.4, 8)],
            [('Invert', 0.6, 4), ('Equalize', 0.6, 7)],
            [('Equalize', 0.6, 7), ('Equalize', 0.6, 7)],
            [('Equalize', 0.6, 7), ('Invert', 0.6, 5)],
            [('Solarize', 0.7, 2), ('Equalize', 0.6, 7)],
            [('Equalize', 0.6, 7), ('Invert', 0.8, 8)],
            [('Equalize', 0.6, 7), ('Equalize', 0.6, 7)],
            [('Equalize', 0.6, 7), ('AutoContrast', 0.7, 3)],
            [('Equalize', 0.6, 7), ('Invert', 0.1, 5)]]


# Unique operations: Invert, Color, Contrast, Sharpness, AutoContrast,
#                    Equalize, Solarize, Posterize, Brightness
def good_policies_cifar():
    """AutoAugment policies found on Cifar."""
    exp0_0 = [
      [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],
      [('Color', 0.2, 4), ('Color', 0.2, 4)],
      [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
      [('Color', 0.2, 4), ('Color', 0.2, 4)],
      [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)]]
    exp0_1 = [
      [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],
      [('Color', 0.2, 4), ('Color', 0.2, 4)],
      [('AutoContrast', 0.9, 2), ('Solarize', 0.8, 3)],
      [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],
      [('Color', 0.2, 4), ('AutoContrast', 0.9, 1)]]
    exp0_2 = [
      [('Solarize', 0.4, 5), ('AutoContrast', 0.0, 2)],
      [('Color', 0.2, 4), ('Color', 0.2, 4)],
      [('AutoContrast', 0.9, 0), ('Solarize', 0.4, 3)],
      [('Equalize', 0.7, 5), ('Invert', 0.1, 3)],
      [('Color', 0.2, 4), ('Color', 0.2, 4)]]
    exp0_3 = [
      [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 1)],
      [('Color', 0.2, 4), ('Color', 0.2, 4)],
      [('AutoContrast', 0.8, 0), ('Color', 0.2, 4)],
      [('Color', 0.2, 4), ('Color', 0.9, 6)],
      [('Equalize', 0.7, 6), ('Color', 0.4, 9)]]
    exp1_0 = [
      [('Color', 0.2, 4), ('Posterize', 0.3, 7)],
      [('Color', 0.4, 3), ('Brightness', 0.6, 7)],
      [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],
      [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],
      [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)]]
    exp1_1 = [
      [('Brightness', 0.3, 7), ('AutoContrast', 0.5, 8)],
      [('AutoContrast', 0.9, 4), ('AutoContrast', 0.5, 6)],
      [('Solarize', 0.3, 5), ('Equalize', 0.6, 5)],
      [('Color', 0.2, 4), ('Sharpness', 0.3, 3)],
      [('Brightness', 0.0, 8), ('Color', 0.8, 8)]]
    exp1_2 = [
      [('Solarize', 0.2, 6), ('Color', 0.8, 6)],
      [('Solarize', 0.2, 6), ('AutoContrast', 0.8, 1)],
      [('Solarize', 0.4, 1), ('Equalize', 0.6, 5)],
      [('Brightness', 0.0, 0), ('Solarize', 0.5, 2)],
      [('AutoContrast', 0.9, 5), ('Brightness', 0.5, 3)]]
    exp1_3 = [
      [('Contrast', 0.7, 5), ('Brightness', 0.0, 2)],
      [('Solarize', 0.2, 8), ('Solarize', 0.1, 5)],
      [('Contrast', 0.5, 1), ('Color', 0.2, 4)],
      [('AutoContrast', 0.6, 5), ('Color', 0.2, 4)],
      [('AutoContrast', 0.9, 4), ('Equalize', 0.8, 4)]]
    exp1_4 = [
      [('Brightness', 0.0, 7), ('Equalize', 0.4, 7)],
      [('Solarize', 0.2, 5), ('Equalize', 0.7, 5)],
      [('Equalize', 0.6, 8), ('Color', 0.6, 2)],
      [('Color', 0.3, 7), ('Color', 0.2, 4)],
      [('AutoContrast', 0.5, 2), ('Solarize', 0.7, 2)]]
    exp1_5 = [
      [('AutoContrast', 0.2, 0), ('Equalize', 0.1, 0)],
      [('Color', 0.2, 4), ('Equalize', 0.6, 5)],
      [('Brightness', 0.9, 3), ('AutoContrast', 0.4, 1)],
      [('Equalize', 0.8, 8), ('Equalize', 0.7, 7)],
      [('Equalize', 0.7, 7), ('Solarize', 0.5, 0)]]
    exp1_6 = [
      [('Equalize', 0.8, 4), ('Color', 0.2, 4)],
      [('Color', 0.2, 4), ('Color', 0.2, 4)],
      [('Color', 0.2, 4), ('Color', 0.2, 4)],
      [('AutoContrast', 0.5, 3), ('Solarize', 0.3, 4)],
      [('Solarize', 0.5, 3), ('Equalize', 0.4, 4)]]
    exp2_0 = [
      [('Color', 0.7, 7), ('Color', 0.2, 4)],
      [('Equalize', 0.3, 7), ('AutoContrast', 0.4, 8)],
      [('Color', 0.2, 4), ('Sharpness', 0.2, 6)],
      [('Brightness', 0.9, 6), ('Color', 0.2, 8)],
      [('Solarize', 0.5, 2), ('Invert', 0.0, 3)]]
    exp2_1 = [
      [('AutoContrast', 0.1, 5), ('Brightness', 0.0, 0)],
      [('Color', 0.2, 4), ('Equalize', 0.1, 1)],
      [('Equalize', 0.7, 7), ('AutoContrast', 0.6, 4)],
      [('Color', 0.1, 8), ('Color', 0.2, 4)],
      [('Color', 0.2, 4), ('Color', 0.2, 4)]]
    exp2_2 = [
      [('Color', 0.2, 4), ('AutoContrast', 0.9, 5)],
      [('Color', 0.2, 4), ('Color', 0.2, 4)],
      [('Equalize', 0.5, 0), ('Solarize', 0.6, 6)],
      [('AutoContrast', 0.3, 5), ('Color', 0.2, 4)],
      [('Equalize', 0.8, 2), ('Invert', 0.4, 0)]]
    exp2_3 = [
      [('Equalize', 0.9, 5), ('Color', 0.7, 0)],
      [('Equalize', 0.1, 1), ('Color', 0.2, 4)],
      [('AutoContrast', 0.7, 3), ('Equalize', 0.7, 0)],
      [('Brightness', 0.5, 1), ('Contrast', 0.1, 7)],
      [('Contrast', 0.1, 4), ('Solarize', 0.6, 5)]]
    exp2_4 = [
      [('Solarize', 0.2, 3), ('Color', 0.2, 4)],
      [('Color', 0.2, 4), ('Color', 0.2, 4)],
      [('Equalize', 0.5, 9), ('Color', 0.2, 4)],
      [('Color', 0.2, 4), ('Sharpness', 0.5, 1)],
      [('Equalize', 0.8, 6), ('Invert', 0.3, 6)]]
    exp2_5 = [
      [('AutoContrast', 0.3, 9), ('Color', 0.2, 4)],
      [('Color', 0.2, 4), ('AutoContrast', 0.9, 2)],
      [('Color', 0.2, 4), ('Posterize', 0.0, 3)],
      [('Solarize', 0.4, 3), ('Color', 0.2, 4)],
      [('Equalize', 0.1, 4), ('Equalize', 0.7, 6)]]
    exp2_6 = [
      [('Equalize', 0.3, 8), ('AutoContrast', 0.4, 3)],
      [('Solarize', 0.6, 4), ('AutoContrast', 0.7, 6)],
      [('AutoContrast', 0.2, 9), ('Brightness', 0.4, 8)],
      [('Equalize', 0.1, 0), ('Equalize', 0.0, 6)],
      [('Equalize', 0.8, 4), ('Equalize', 0.0, 4)]]
    exp2_7 = [
      [('Equalize', 0.5, 5), ('AutoContrast', 0.1, 2)],
      [('Solarize', 0.5, 5), ('AutoContrast', 0.9, 5)],
      [('AutoContrast', 0.6, 1), ('AutoContrast', 0.7, 8)],
      [('Equalize', 0.2, 0), ('AutoContrast', 0.1, 2)],
      [('Equalize', 0.6, 9), ('Equalize', 0.4, 4)]]
    exp0s = exp0_0 + exp0_1 + exp0_2 + exp0_3
    exp1s = exp1_0 + exp1_1 + exp1_2 + exp1_3 + exp1_4 + exp1_5 + exp1_6
    exp2s = exp2_0 + exp2_1 + exp2_2 + exp2_3 + exp2_4 + exp2_5 + exp2_6 + exp2_7
    return exp0s + exp1s + exp2s
