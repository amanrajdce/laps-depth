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
"""Data utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
try:
    import cPickle as pickle
except:
    import pickle
import os
import random
import numpy as np
import tensorflow as tf
import cv2
import torch
import torch.utils.data as data

import pba.policies as found_policies
from pba.utils import parse_log_schedule
import pba.augmentation_transforms_hp as augmentation_transforms_pba
import pba.augmentation_transforms as augmentation_transforms_autoaug


class PairedData(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        tgt_img, src_img_1, src_img_2, intrinsic = next(self.data_loader_iter)
        return tgt_img, src_img_1, src_img_2, intrinsic


class ImageFolderKITTI(data.Dataset):
    def __init__(
        self, data_root, train_file_path, input_height, input_width,
    ):
        """
        :param data_root: dataset root
        :param train_file_path: path to train file
        :param input_height: height of input image
        :param input_width: width of input image
        """
        # load image and camera params file list
        self.data_root = data_root
        self.train_file_path = train_file_path
        self.input_height = input_height
        self.input_width = input_width
        self.data = self.format_file_list()
        self.image_list = self.data['image_file_list']
        self.cam_list = self.data['cam_file_list']
        assert len(self.image_list) == len(self.cam_list)

    def read_image_data_for_input(self, img_path):
        """
        Read image sequence and split frames, normalize to range [0, 1]
        :param img_path: path of image sequence
        :return: tgt_img, src_img_1, srcm_2
        """
        # read image, convert to RGB
        image_seq = cv2.imread(img_path)
        image_seq = cv2.cvtColor(image_seq, cv2.COLOR_BGR2RGB)
        src_img_1 = image_seq[:, :self.input_width, :]
        tgt_img = image_seq[:, self.input_width:2*self.input_width, :]
        src_img_2 = image_seq[:, 2*self.input_width:, :]

        # Normalize
        tgt_img = tgt_img.astype('float32') / 255.0
        src_img_1 = src_img_1.astype('float32') / 255.0
        src_img_2 = src_img_2.astype('float32') / 255.0

        return tgt_img, src_img_1, src_img_2

    def format_file_list(self):
        """
        Read the list of image files and intrinsics camera params
        :return: dictionary of read image and cam files
        """
        with open(self.train_file_path, 'r') as f:
            frames = f.readlines()

        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [
            os.path.join(self.data_root, subfolders[i], frame_ids[i] + '.png') for i in range(len(frames))
        ]
        cam_file_list = [
            os.path.join(self.data_root, subfolders[i], frame_ids[i] + '_cam.txt') for i in range(len(frames))
        ]

        all_list = {'image_file_list': image_file_list, 'cam_file_list': cam_file_list}

        return all_list

    def read_cam_data(self, cam_path):
        """
        :param cam_path: path to camera intrinsic .txt
        :return: 3x3 camera intrinsic
        """
        with open(cam_path, 'r') as f:
            lines = f.readlines()

        intrinsic = np.asarray(lines[0].split(",")).astype('float32')
        intrinsic = np.reshape(intrinsic, [3, 3])

        return intrinsic

    def __getitem__(self, index):
        img_path = self.image_list[index]
        cam_path = self.cam_list[index]
        tgt_img, src_img_1, src_img_2 = self.read_image_data_for_input(img_path)
        intrinsic = self.read_cam_data(cam_path)

        return tgt_img, src_img_1, src_img_2, intrinsic

    def __len__(self):
        return len(self.image_list)


def parse_policy(policy_emb, augmentation_transforms):
    policy = []
    num_xform = augmentation_transforms.NUM_HP_TRANSFORM
    xform_names = augmentation_transforms.HP_TRANSFORM_NAMES
    assert len(policy_emb) == 2 * num_xform, 'policy was: {}, supposed to be: {}'.format(len(policy_emb), 2 * num_xform)

    for i, xform in enumerate(xform_names):
        policy.append((xform, policy_emb[2 * i] / 10., policy_emb[2 * i + 1]))
    return policy


class TrainDataSet(object):
    """Dataset object that produces augmented training data"""

    def __init__(self, hparams, shuffle=False):
        """
        :param hparams: tf.hparams object
        """
        self.hparams = hparams
        self.epochs = 0
        seed = 8964
        random.seed(seed)
        self.input_height = hparams.input_height
        self.input_width = hparams.input_width

        # parsing initial policy for data augmentation
        self.parse_policy(hparams)
        dataset = ImageFolderKITTI(
            self.hparams.kitti_root, self.hparams.train_file_path,
            self.hparams.input_height, self.hparams.input_width
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.hparams.batch_size, shuffle=shuffle,
            num_workers=self.hparams.num_workers, drop_last=True
        )
        self.train_size = len(dataset)
        self.paired_data = PairedData(data_loader)
        tf.logging.info('Train dataset size: {}'.format(len(dataset)))

    def parse_policy(self, hparams):
        """Parses policy schedule from input, which can be a list, list of lists, text file, or pickled list.

        If list is not nested, then uses the same policy for all epochs.

        Args:
        hparams: tf.hparams object.
        """
        if hparams.no_aug_policy:
            tf.logging.info("no augmentation policy will be used")
            if hparams.use_kitti_aug:
                tf.logging.info("using augmentations from SIGNet")
            return

        # Parse policy
        if hparams.use_hp_policy:
            self.augmentation_transforms = augmentation_transforms_pba
            tf.logging.info('hp policy is selected')
            if isinstance(hparams.hp_policy, str) and hparams.hp_policy.endswith('.txt'):
                if hparams.num_epochs % hparams.hp_policy_epochs != 0:
                    tf.logging.warning(
                        "Schedule length (%s) doesn't divide evenly into epochs (%s), interpolating.",
                        hparams.num_epochs, hparams.hp_policy_epochs
                    )
                tf.logging.info(
                    'schedule policy trained on {} epochs, parsing from: {}, multiplier: {}'.format(
                        hparams.hp_policy_epochs, hparams.hp_policy,
                        float(hparams.num_epochs) / hparams.hp_policy_epochs
                    )
                )
                raw_policy = parse_log_schedule(
                    hparams.hp_policy,
                    epochs=hparams.hp_policy_epochs,
                    multiplier=float(hparams.num_epochs) / hparams.hp_policy_epochs
                )
            elif isinstance(hparams.hp_policy, str) and hparams.hp_policy.endswith('.p'):
                assert hparams.num_epochs % hparams.hp_policy_epochs == 0
                tf.logging.info('custom .p file, policy number: {}'.format(hparams.schedule_num))
                with open(hparams.hp_policy, 'rb') as f:
                    policy = pickle.load(f)[hparams.schedule_num]
                raw_policy = []
                for num_iters, pol in policy:
                    for _ in range(num_iters * hparams.num_epochs // hparams.hp_policy_epochs):
                        raw_policy.append(pol)
            else:
                raw_policy = hparams.hp_policy

            if isinstance(raw_policy[0], list):
                self.policy = []
                split = len(raw_policy[0]) // 2
                for pol in raw_policy:
                    cur_pol = parse_policy(pol[:split], self.augmentation_transforms)
                    cur_pol.extend(parse_policy(pol[split:], self.augmentation_transforms))
                    self.policy.append(cur_pol)
                tf.logging.info('using HP policy schedule, last: {}'.format(self.policy[-1]))
            elif isinstance(raw_policy, list):
                split = len(raw_policy) // 2
                self.policy = parse_policy(raw_policy[:split], self.augmentation_transforms)
                self.policy.extend(parse_policy(raw_policy[split:], self.augmentation_transforms))
                tf.logging.info('using HP Policy, policy: {}'.format(self.policy))
        else:
            # use autoaugment policies modified for KITTI
            self.augmentation_transforms = augmentation_transforms_autoaug
            tf.logging.info('using autoaument policy: {}'.format(hparams.policy_dataset))
            if hparams.policy_dataset == 'svhn':
                self.good_policies = found_policies.good_policies_svhn()
            else:
                # use cifar10 good policies
                self.good_policies = found_policies.good_policies_cifar()

    def reset_policy(self, new_hparams):
        self.hparams = new_hparams
        self.parse_policy(new_hparams)
        tf.logging.info('reset aug policy')
        return

    def augment_batch(self, tgt_img_batch, src_img_1_batch, src_img_2_batch, intrinsic_batch, iteration=None):
        """
        Apply augmentation on given batch of data
        :param tgt_img_batch:  target image batch
        :param src_img_1_batch: srch_image1 batch
        :param src_img_2_batch: src_image2 batch
        :param intrinsic_batch: intrinsic matrix batch
        :param iteration: current iteration number
        :return: augmented batch
        """
        # convert pytorch tensors back to numpy
        tgt_img_batch = tgt_img_batch.numpy()
        src_img_1_batch = src_img_1_batch.numpy()
        src_img_2_batch = src_img_2_batch.numpy()
        intrinsic_batch = intrinsic_batch.numpy()

        tgt_img_batch_aug = []
        src_img_1_batch_aug = []
        src_img_2_batch_aug = []
        intrinsic_batch_aug = []

        assert len(tgt_img_batch) == len(src_img_1_batch) == len(src_img_2_batch) == len(intrinsic_batch)
        curr_batch_size = len(tgt_img_batch)

        for idx in range(0, curr_batch_size):
            tgt_img = tgt_img_batch[idx]
            src_img_1 = src_img_1_batch[idx]
            src_img_2 = src_img_2_batch[idx]
            intrinsic = intrinsic_batch[idx]
            if not self.hparams.no_aug_policy:
                if not self.hparams.use_hp_policy:
                    # apply autoaugment policy here modified for KITTI
                    epoch_policy = self.good_policies[np.random.choice(len(self.good_policies))]
                    tgt_img, src_img_1, src_img_2, intrinsic = self.augmentation_transforms.apply_policy(
                        policy=epoch_policy, data=[tgt_img, src_img_1, src_img_2, intrinsic],
                        image_size=(self.input_height, self.input_width)
                    )
                else:
                    # apply PBA policy modified for KITTI
                    if isinstance(self.policy[0], list):
                        # single policy
                        if self.hparams.flatten:
                            tgt_img, src_img_1, src_img_2, intrinsic = self.augmentation_transforms.apply_policy(
                                policy=self.policy[random.randint(0, len(self.policy) - 1)],
                                data=[tgt_img, src_img_1, src_img_2, intrinsic],
                                image_size=(self.input_height, self.input_width)
                            )
                        else:
                            tgt_img, src_img_1, src_img_2, intrinsic = self.augmentation_transforms.apply_policy(
                                policy=self.policy[iteration],
                                data=[tgt_img, src_img_1, src_img_2, intrinsic],
                                image_size=(self.input_height, self.input_width)
                            )
                    elif isinstance(self.policy, list):
                        # policy schedule
                        tgt_img, src_img_1, src_img_2, intrinsic = self.augmentation_transforms.apply_policy(
                            policy=self.policy, data=[tgt_img, src_img_1, src_img_2, intrinsic],
                            image_size=(self.input_height, self.input_width)
                        )
                    else:
                        raise ValueError('Unknown policy.')
            else:
                # no data augmentation policy
                pass

            if self.hparams.use_kitti_aug:
                # TODO implement augmentations from SIGNet
                # random_scaling, random_cropping,  random_coloring
                raise NotImplementedError()

            tgt_img_batch_aug.append(tgt_img)
            src_img_1_batch_aug.append(src_img_1)
            src_img_2_batch_aug.append(src_img_2)
            intrinsic_batch_aug.append(intrinsic)

        tgt_img_batch = np.array(tgt_img_batch_aug, np.float32)
        src_img_1_batch = np.array(src_img_1_batch_aug, np.float32)
        src_img_2_batch = np.array(src_img_2_batch_aug, np.float32)
        intrinsic_batch = np.array(intrinsic_batch, np.float32)

        # convert source image into (B, H, W, 3*2)
        src_img_stack_batch = np.concatenate((src_img_1_batch, src_img_2_batch), axis=-1)

        return tgt_img_batch, src_img_stack_batch, intrinsic_batch

    def load_data(self):
        return self.paired_data


