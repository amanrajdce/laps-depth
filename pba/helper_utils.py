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
"""Helper functions used for training PBA models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import PIL.Image as pil
import cv2
import time


def compute_errors(gt, pred):
    if np.size(gt) == 0:
        return 0, 0, 0, 0, 1, 1, 1

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def eval_child_model(session, model, test_files):
    """
    Evaluates `model` on held out data depending on `mode`.

    Args:
        session: TensorFlow session the model will be run with.
        model: TensorFlow model that will be evaluated.
        test_files: list of testing files

    Returns:
        Accuracy of `model` when evaluated on the specified dataset.

    Raises:
        ValueError: if invalid dataset `mode` is specified.
    """
    batch_size = model.test_batch_size
    input_height = model.input_height
    input_width = model.input_width
    tf.logging.info('model.test_batch_size is {}'.format(batch_size))

    preds_all = []
    for t in range(0, len(test_files), batch_size):
        inputs = np.zeros((batch_size, input_height, input_width, 3))
        # batching for evaluation
        for b in range(batch_size):
            idx = t + b
            if idx >= len(test_files):
                break
            # adapt to py3 ref: https://github.com/python-pillow/Pillow/issues/1605
            fh = open(test_files[idx], 'rb')
            raw_im = pil.open(fh)
            scaled_im = raw_im.resize((input_width, input_height), pil.ANTIALIAS)
            inputs[b] = np.array(scaled_im)

        preds = session.run(
            model.pred_depth,
            feed_dict={model.tgt_image_input: inputs}
        )
        for b in range(batch_size):
            idx = t + b
            if idx >= len(test_files):
                break
            preds_all.append(preds[b, :, :, 0])

    assert len(preds_all) == len(test_files)

    return preds_all


def run_evaluation(gt_depths, pred_depths, min_depth=1e-3, max_depth=80, verbose=False):
    t1 = time.time()
    num_test = len(gt_depths)
    pred_depths_resized = []

    for t_id in range(num_test):
        img_size_h, img_size_w = gt_depths[t_id].shape
        pred_depths_resized.append(
            cv2.resize(pred_depths[t_id], (img_size_w, img_size_h), interpolation=cv2.INTER_LINEAR)
        )

    pred_depths = pred_depths_resized
    rms = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel = np.zeros(num_test, np.float32)
    d1_all = np.zeros(num_test, np.float32)
    a1 = np.zeros(num_test, np.float32)
    a2 = np.zeros(num_test, np.float32)
    a3 = np.zeros(num_test, np.float32)

    for i in range(num_test):
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])

        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
        # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        gt_height, gt_width = gt_depth.shape

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        # Scale matching
        scalar = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
        pred_depth[mask] *= scalar

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(
            gt_depth[mask], pred_depth[mask]
        )

    if verbose:
        tf.logging.info('Evaluating for errors took {:.4f} secs'.format(time.time() - t1))  # adapt to py3
        tf.logging.info("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
            'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3')
        )
        tf.logging.info("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(
            abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean())
        )

    return abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()


def cosine_lr(learning_rate, epoch, iteration, batches_per_epoch, total_epochs):
    """
    Cosine Learning rate.

    Args:
        learning_rate: Initial learning rate.
        epoch: Current epoch we are one. This is one based.
        iteration: Current batch in this epoch.
        batches_per_epoch: Batches per epoch.
        total_epochs: Total epochs you are training for.

    Returns:
        The learning rate to be used for this current batch.
    """
    t_total = total_epochs * batches_per_epoch
    t_cur = float(epoch * batches_per_epoch + iteration)
    return 0.5 * learning_rate * (1 + np.cos(np.pi * t_cur / t_total))


# TODO might have to tune decay rate for KITTI
def step_lr(learning_rate, epoch):
    """
    Step Learning rate.

    Args:
        learning_rate: Initial learning rate.
        epoch: Current epoch we are one. This is one based.

    Returns:
        The learning rate to be used for this current batch.
    """
    if epoch < 80:
        return learning_rate
    elif epoch < 120:
        return learning_rate * 0.1
    else:
        return learning_rate * 0.01


def get_lr(curr_epoch, hparams, train_size, iteration=None):
    """Returns the learning rate during training based on the current epoch."""
    assert iteration is not None
    batches_per_epoch = int(train_size / hparams.batch_size)

    # No lr decay enabled
    if hparams.lr_decay is None:
        return hparams.lr

    if hparams.lr_decay == "step":
        lr = step_lr(hparams.lr, curr_epoch)
    elif hparams.lr_decay == "cosine":
        lr = cosine_lr(
            hparams.lr, curr_epoch, iteration, batches_per_epoch, hparams.num_epochs
        )
        tf.logging.log_first_n(tf.logging.WARN, 'Default to cosine learning rate.', 1)
    else:
        raise ValueError("Unknown  lr decay policy!!")

    return lr


def run_epoch_training(session, model, data_loader, train_size, batch_aug_fn, curr_epoch):
    """Runs one epoch of training for the model passed in.

    Args:
        session: TensorFlow session the model will be run with.
        model: TensorFlow model that will be evaluated.
        data_loader: DataSet object that contains data that `model` will evaluate.
        curr_epoch: How many of epochs of training have been done so far.
        train_size: Size of training set

    Returns:
        The accuracy of 'model' on the training set
    """
    steps_per_epoch = int(train_size / model.hparams.batch_size)
    tf.logging.info('steps per epoch: {}'.format(steps_per_epoch))
    curr_step = session.run(model.global_step)
    assert curr_step % steps_per_epoch == 0

    # Get the current learning rate for the model based on the current epoch
    curr_lr = get_lr(curr_epoch, model.hparams, train_size, iteration=0)
    tf.logging.info('lr of {} for epoch {}'.format(curr_lr, curr_epoch))

    # train for one epoch
    for step, batch in enumerate(data_loader):
        curr_lr = get_lr(curr_epoch, model.hparams, train_size, iteration=(step + 1))
        # Update the lr rate variable to the current LR.
        model.lr_rate_ph.load(curr_lr, session=session)
        if step % 20 == 0:
            tf.logging.info('Training {}/{}'.format(step, steps_per_epoch))

        tgt_img, src_img_1, src_img_2, intrinsic = batch
        tgt_image, src_image_stack, intrinsic = batch_aug_fn(tgt_img, src_img_1, src_img_2, intrinsic, curr_epoch)

        _, step, preds = session.run(
            [model.train_op, model.global_step, model.pred_depth],
            feed_dict={
                model.tgt_image_input: tgt_image,
                model.src_image_stack_input: src_image_stack,
                model.intrinsic_input: intrinsic
            })

