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
"""
PBA & AutoAugment Train/Eval module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import time

import numpy as np
import tensorflow as tf
import gc

import pba.data_utils as data_utils
import pba.helper_utils as helper_utils
from pba.sig_model import Model


class ModelTrainer(object):
    """Trains an instance of the Model class."""

    def __init__(self, hparams, comet_exp=None):
        """
        Build model architecture and setup train and test dataloaders
        :param hparams: parsed params
        :param comet_exp: name of comet project to log data online
        """
        self._session = None
        self.hparams = hparams
        self.comet_exp = comet_exp

        self.dataset = data_utils.TrainDataSet(hparams)
        self.train_size = self.dataset.train_size
        self.data_loader = self.dataset.load_data()

        # Loading gt data and files for test set
        self.test_files = self.read_test_files(self.hparams.kitti_raw)
        self.gt_depths = self.setup_evaluation(self.hparams.gt_path)
        self.hparams.add_hparam('train_size', self.train_size)

        if self.comet_exp is not None:
            self.comet_exp.log_parameters(self.hparams.values())

        # extra stuff for ray
        self._build_models()
        self._new_session()
        self._session.__enter__()

    def read_test_files(self, dataset_dir):
        """
        Read test files for depth estimation
        :param dataset_dir: root path of kitti dataset
        :return: read test files with kitti root path added as prefix
        """
        with open(self.hparams.test_file_path, 'r') as f:
            test_files = f.readlines()
            test_files = [t.rstrip() for t in test_files]
            test_files = [os.path.join(dataset_dir, t) for t in test_files]

        return test_files

    def save_model(self, checkpoint_dir, step=None):
        """Dumps model into the backup_dir.

        Args:
          step: If provided, creates a checkpoint with the given step
            number, instead of overwriting the existing checkpoints.
        """
        model_save_name = os.path.join(checkpoint_dir, 'model.ckpt') + '-' + str(step)
        save_path = self.saver.save(self.session, model_save_name)
        tf.logging.info('Saved child model')

        return model_save_name

    def extract_model_spec(self, checkpoint_path):
        """Loads a checkpoint with the architecture structure stored in the name."""
        self.saver.restore(self.session, checkpoint_path)
        tf.logging.warning('Loaded child model checkpoint from {}'.format(checkpoint_path))

    def eval_child_model(self, model, epoch):
        """Evaluate the child model.

        Args:
          model: image model that will be evaluated.
          epoch: epoch number on which model is evaluated

        Returns:
          predicted depth maps on kitti test split
        """
        tf.logging.info('Evaluating child model')
        while True:
            try:
                preds_all = helper_utils.eval_child_model(self.session, model, epoch, self.test_files, self.comet_exp)
                # If epoch trained without raising the below errors, break
                # from loop.
                break
            except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
                tf.logging.info('Retryable error caught: {}.  Retrying.'.format(e))

        return preds_all

    @contextlib.contextmanager
    def _new_session(self):
        """Creates a new session for model m."""
        # Create a new session for this model, initialize
        # variables, and save / restore from checkpoint.
        sess_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_cfg.gpu_options.allow_growth = True
        self._session = tf.Session('', config=sess_cfg)
        self._session.run([self.m.init, self.meval.init])

        # log the graph to comet.ml
        if self.comet_exp is not None:
            self.comet_exp.set_model_graph(self._session.graph)
        return self._session

    def _build_models(self):
        """Builds the image models for train and eval."""
        # Determine if we should build the train and eval model. When using
        # distributed training we only want to build one or the other and not both.
        with tf.variable_scope('model', use_resource=False):
            m = Model(self.hparams, mode='train')
            m.build()
            self._num_trainable_params = m.num_trainable_params
            self._saver = m.saver
        with tf.variable_scope('model', reuse=True, use_resource=False):
            meval = Model(self.hparams, mode='eval')
            meval.build()
        self.m = m
        self.meval = meval

    def _run_training_loop(self, curr_epoch):
        """Trains the model `m` for one epoch."""
        # free up ray memory
        gc.collect()

        start_time = time.time()
        while True:
            try:
                helper_utils.run_epoch_training(
                    self.session, self.m, self.dataset, self.train_size, curr_epoch, self.comet_exp
                )
                break
            except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
                tf.logging.info('Retryable error caught: {}.  Retrying.'.format(e))
        tf.logging.info('Finished epoch: {}'.format(curr_epoch))
        tf.logging.info('Epoch time(min): {}'.format((time.time() - start_time) / 60.0))

    def run_model(self, epoch):
        """Trains and evalutes the image model."""
        self._run_training_loop(epoch)
        eval_preds = self.eval_child_model(self.meval, epoch)
        return eval_preds

    def setup_evaluation(self, gt_path):
        """
        Read ground truth depth information for test set
        :param gt_path: path to .npy file containing gt depth
        :return: gt depth data for test set in numpy array form
        """
        # self.gt_path: /path/to/gt_depth.npy or /path/to/gt_interp_depth.npy

        loaded_gt_depths = np.load(gt_path)
        num_test = len(loaded_gt_depths)
        gt_depths = []

        for t_id in range(num_test):
            depth = loaded_gt_depths[t_id]
            gt_depths.append(depth.astype(np.float32))

        return gt_depths

    def run_evaluation(self, pred_depths, epoch, verbose=True):
        results = helper_utils.run_evaluation(
            self.gt_depths, pred_depths, epoch, min_depth=self.hparams.min_depth,
            max_depth=self.hparams.max_depth, verbose=verbose, comet_exp=self.comet_exp
        )
        return results

    def reset_config(self, new_hparams):
        self.hparams = new_hparams
        self.dataset.reset_policy(new_hparams)
        return

    @property
    def saver(self):
        return self._saver

    @property
    def session(self):
        return self._session

    @property
    def num_trainable_params(self):
        return self._num_trainable_params
