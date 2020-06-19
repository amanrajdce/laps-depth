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
import json as js
import os

import gc


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


def compute_predictions(session, model, global_step, test_files, comet_exp=None):
    """
    Compute predictions for evaluation model.

    Args:
        session: TensorFlow session the model will be run with.
        model: TensorFlow model that will be evaluated.
        global_step: current global step of model
        test_files: list of testing files
        comet_exp: comet.ml experiment name to write log to

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
            raw_im = pil.open(fh).convert('RGB')
            scaled_im = raw_im.resize((input_width, input_height), pil.ANTIALIAS)
            inputs[b] = np.array(scaled_im, dtype='float32') / 255.0

        preds = session.run(
            model.pred_depth[0],
            feed_dict={model.tgt_image_input: inputs}
        )
        for b in range(batch_size):
            idx = t + b
            if idx >= len(test_files):
                break
            preds_all.append(preds[b, :, :, 0])

        # comet.ml log all the images and predictions
        if comet_exp is not None and t % 40 == 0:
            curr_batch_size = len(inputs)
            with comet_exp.test():
                for b in range(0, curr_batch_size):
                    step = t + b
                    # input image
                    comet_exp.log_image(
                        inputs[b, :, :, :], name="test_iter" + str(step) + "_input",
                        image_format="png", image_channels="last", step=global_step
                    )
                    # prediction
                    comet_exp.log_image(
                        preds[b, :, :, :], name="test_iter" + str(step) + "_pred", image_format="png",
                        image_colormap="plasma", image_channels="last", step=global_step
                    )

    assert len(preds_all) == len(test_files)

    return preds_all


def eval_predictions(gt_depths, pred_depths, global_step, min_depth=1e-3, max_depth=80, verbose=True, comet_exp=None):
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

    # compute mean of all errors and accuracies
    abs_rel = abs_rel.mean()
    sq_rel = sq_rel.mean()
    rms = rms.mean()
    log_rms = log_rms.mean()
    d1_all = d1_all.mean()
    a1 = a1.mean()
    a2 = a2.mean()
    a3 = a3.mean()
    abs_rel_acc = 0. if abs_rel > 1. else 1. - abs_rel

    if verbose:
        tf.logging.info('Evaluating for errors took {:.4f} secs'.format(time.time() - t1))  # adapt to py3
        tf.logging.info("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
            'abs_rel_acc', 'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3')
        )
        tf.logging.info("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(
            abs_rel_acc, abs_rel, sq_rel, rms, log_rms, d1_all, a1, a2, a3)
        )

    results = {
        "abs_rel": abs_rel,
        "abs_rel_acc": abs_rel_acc,
        "sq_rel": sq_rel,
        "rms": rms,
        "log_rms": log_rms,
        "d1_all": d1_all,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "global_step": global_step  # training_iteration is reserved by ray to keep track of epoch
    }
    if comet_exp is not None:
        with comet_exp.test():  # to provide context for comet.ml graphs
            comet_exp.log_metrics(results, step=global_step)

    return results


def cosine_lr(learning_rate, cur_epoch, iteration, steps_per_epoch, total_epochs):
    """
    Cosine Learning rate.

    Args:
        learning_rate: Initial learning rate.
        cur_epoch: Current epoch we are one. This is 1 based.
        iteration: Current batch in this epoch.
        steps_per_epoch: number of steps per epoch
        total_epochs: Total epochs you are training for.

    Returns:
        The learning rate to be used for current step.
    """
    total_steps = total_epochs * steps_per_epoch
    cur_step = float((cur_epoch-1) * steps_per_epoch + iteration)
    return 0.5 * learning_rate * (1 + np.cos(np.pi * cur_step / total_steps))


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
    if epoch < 16:
        return learning_rate
    elif epoch < 31:
        return learning_rate * 0.1
    else:
        return learning_rate * 0.01


def get_lr(curr_epoch, hparams, steps_per_epoch, iteration=None):
    """Returns the learning rate during training based on the current epoch."""
    assert iteration is not None

    # No lr decay enabled
    if hparams.lr_decay is None:
        return hparams.lr

    if hparams.lr_decay == "step":
        lr = step_lr(hparams.lr, curr_epoch)
    elif hparams.lr_decay == "cosine":
        lr = cosine_lr(
            hparams.lr, curr_epoch, iteration, steps_per_epoch, hparams.num_epochs
        )
    else:
        raise ValueError("Unknown lr decay policy!!")

    return lr


def run_epoch_training(
        session, model, dataset, train_size, curr_epoch, comet_exp=None,
        last_ckpt_dir=None, model_saver=None, eval_fun=None
):
    """Runs one epoch of training for the model passed in.

    Args:
        session: TensorFlow session the model will be run with.
        model: TensorFlow model that will be evaluated.
        dataset: DataSet object that contains train data that `model` will train on
        curr_epoch: current epoch of model training
        train_size: Size of training set
        comet_exp: comet.ml experiment name to write log to
        last_ckpt_dir: directory of last saved ckpt by ray
        model_saver: tf.Saver object
        eval_fun: function that predicts and evaluates

    Returns:
        The accuracy of 'model' on the training set
    """
    global_seed = 8964
    batch_size = model.hparams.batch_size

    steps_per_epoch = int(train_size / batch_size)
    assert steps_per_epoch == model.hparams.steps_per_epoch
    tf.logging.info('steps per epoch: {}'.format(steps_per_epoch))
    curr_step = session.run(model.global_step)
    tf.logging.info("Current step: {}".format(curr_step))
    # assert curr_step % steps_per_epoch == 0

    # Get the current learning rate for the model based on the current epoch
    curr_lr = get_lr(curr_epoch, model.hparams, train_size, iteration=0)
    tf.logging.info('lr of {} for epoch {}'.format(curr_lr, curr_epoch))

    # shuffle training indexes to form batch
    # drop last indexes that dont form full batch
    np.random.seed(global_seed + curr_epoch)
    train_indxs = list(range(0, train_size))
    np.random.shuffle(train_indxs)
    train_indxs = train_indxs[: steps_per_epoch*batch_size]

    # train for one epoch
    for step in range(0, steps_per_epoch):
        # free up ray memory
        gc.collect()

        curr_lr = get_lr(curr_epoch, model.hparams, train_size, iteration=(step + 1))
        # Update the lr rate variable to the current LR.
        model.lr_rate_ph.load(curr_lr, session=session)
        if step % model.hparams.log_iter == 0:
            tf.logging.info('Training {}/{}'.format(step, steps_per_epoch))

        # get batch_indexes
        start_idx = step*batch_size
        end_idx = start_idx + batch_size
        batch_indxs = train_indxs[start_idx: end_idx]

        # get original batch as well as augmented from policy methods
        tgt_img, src_img_stack, tgt_img_aug, src_img_stack_aug, intrinsic = dataset.next_batch(batch_indxs, curr_epoch)

        _, global_step, preds, tgt_img_final, src_img_stack_final, fwd_warp, fwd_error, bwd_warp, bwd_error, loss = \
            session.run(
                [model.train_op, model.global_step, model.pred_depth[0],
                 model.tgt_image, model.src_image_stack,
                 model.fwd_rigid_warp_pyramid[0], model.fwd_rigid_error_scale[0],
                 model.bwd_rigid_warp_pyramid[0], model.bwd_rigid_error_scale[0], model.total_loss],
                feed_dict={
                    model.tgt_image_input: tgt_img_aug,
                    model.src_image_stack_input: src_img_stack_aug,
                    model.intrinsic_input: intrinsic
                })

        # tgt_img_final: final target with signet augmentation if enabled
        # src_img_stack_final: final src images with signet augmentation if enabled

        # comet.ml log all the images and errors
        if comet_exp is not None and global_step % model.hparams.log_iter == 0:
            with comet_exp.train():  # train context for comet.ml logging into cloud
                # train metrics
                comet_exp.log_metric("total_loss", loss, step=global_step)
                comet_exp.log_metric("lr", curr_lr, step=global_step)

                # images
                for b in range(0, model.hparams.max_outputs):
                    # create src1_tgt_src2 image
                    src1_tgt_src2 = np.concatenate(
                        (src_img_stack[b, :, :, :3], tgt_img[b, :, :, :], src_img_stack[b, :, :, 3:]), axis=1
                    )
                    src1_tgt_src2_aug = np.concatenate(
                        (src_img_stack_final[b, :, :, :3], tgt_img_final[b, :, :, :], src_img_stack_final[b, :, :, 3:]), axis=1
                    )
                    comet_exp.log_image(
                        src1_tgt_src2, name="src_tgt_src_b" + str(b),
                        image_format="png", image_channels="last", step=global_step
                    )
                    comet_exp.log_image(
                        src1_tgt_src2_aug, name="src_tgt_src_aug_b" + str(b),
                        image_format="png", image_channels="last", step=global_step
                    )
                    # prediction
                    comet_exp.log_image(
                        preds[b, :, :, :], name="preds_b" + str(b), image_format="png",
                        image_colormap="plasma", image_channels="last", step=global_step
                    )
                    # warping results
                    fwd_bwd_warp = np.concatenate(
                        (fwd_warp[b, :, :, :], bwd_warp[b, :, :, :]), axis=1
                    )
                    comet_exp.log_image(
                        fwd_bwd_warp, name="fwd_bwd_warp_b" + str(b),
                        image_format="png", image_channels="last", step=global_step
                    )
                    # error
                    fwd_bwd_warp_error = np.concatenate(
                        (fwd_error[b, :, :, :], bwd_error[b, :, :, :]), axis=1
                    )
                    comet_exp.log_image(
                        fwd_bwd_warp_error, name="fwd_bwd_error_b" + str(b),
                        image_format="png", image_channels="last", step=global_step
                    )

        # checkpoint model at current global step if enabled
        if last_ckpt_dir is not None and model.hparams.checkpoint_iter and \
           curr_epoch > model.hparams.checkpoint_iter_after:
            if global_step % model.hparams.checkpoint_iter == 0:
                last_ckpt_epoch = last_ckpt_dir.split("/")[-1]
                train_dir = last_ckpt_dir.replace(last_ckpt_epoch, "")
                ckpt_dir_itr = os.path.join(train_dir, "checkpoint_itr" + str(global_step))

                assert model_saver is not None, "model saver object is None"
                assert eval_fun is not None, "evaluation fun is None"

                model_save_name = os.path.join(ckpt_dir_itr, 'model.ckpt') + '-' + str(global_step)
                save_path = model_saver.save(session, model_save_name)
                tf.logging.info('Saved child model at:{}'.format(model_save_name))
                os.close(os.open(model_save_name, os.O_CREAT))
                results = eval_fun(epoch=curr_epoch, step=global_step)
                results.update({"training_iteration": curr_epoch})
                results.update(model.hparams.values())

                with open(os.path.join(train_dir, 'result.json'), 'a') as f:
                    f.write(str(results)+"\n")

                if comet_exp is not None:
                    comet_exp.log_parameter("ray_train_dir", train_dir)
