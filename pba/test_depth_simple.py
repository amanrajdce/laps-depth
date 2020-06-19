"""
Evaluates on KITTI eigen test split
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
import cv2
import numpy as np
import PIL.Image as pil
import tensorflow as tf
from pba.model import ModelTrainer
from pba.sig_model import Model
import pba.helper_utils as helper_utils

import matplotlib as mpl
import matplotlib.cm as cm
# mpl.use('Agg')
# import matplotlib.pyplot as plt


class ModelTester(ModelTrainer):
    def __init__(self, hparams, comet_exp=None, log=None):
        self.hparams = hparams
        self.logger = log
        # Loading gt data and files for test set
        self.read_test_files()
        self.comet_exp = comet_exp
        self._build_models()
        self._new_session()
        self._session.__enter__()

    def _build_models(self):
        """Build model for testing"""
        with tf.variable_scope('model'):
            meval = Model(self.hparams, mode='test')
            meval.build()
            self._saver = meval.saver

        self.meval = meval

    def _new_session(self):
        """Creates a new session for model m."""
        # Create a new session for this model, initialize
        # variables, and save / restore from checkpoint.
        sess_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_cfg.gpu_options.allow_growth = True
        self._session = tf.Session('', config=sess_cfg)
        self._session.run([self.meval.init])

        return self._session

    def read_test_files(self):
        """Remove test files """
        if os.path.isfile(self.hparams.image_path):
            images = [self.hparams.image_path]
        else:
            images = os.listdir(self.hparams.image_path)
            images.sort()
            images = [os.path.join(self.hparams.image_path, img) for img in images]
            images = [img for img in images if os.path.isfile(img)]

        self.logger.info("Read files: {}".format(images))
        self.test_files = images

    def run_evaluation(self, epoch=0, global_step=0, verbose=True):
        """Evaluate the child model.

        Args:
          epoch: epoch number at which to evaluate
          global_step: step number at which to evaluate
          verbose: whether to print results

        Returns:
          results dictionary with error and accuracy metrics
        """

        # step-1, compute predictions on test images
        while True:
            try:
                preds_all = helper_utils.compute_predictions(
                    self.session, self.meval, global_step, self.test_files, self.comet_exp
                )
                # If epoch trained without raising the below errors, break from loop.
                break
            except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
                tf.logging.info('Retryable error caught: {}.  Retrying.'.format(e))

        return preds_all

    def save_predictions(self, preds_all, save_dir, scale_pred=False):
        """
        Save predictions and image on disk
        :param preds_all: predicts from model
        :param scale_pred: if enabled scale predictions to min and max depth
        :return:
        """
        for idx, fname in enumerate(self.test_files):
            fh = open(fname, 'rb')
            img = pil.open(fh).convert('RGB')
            orig_h, orig_w, _ = np.array(img).shape
            pred_resize = cv2.resize(preds_all[idx], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            if scale_pred:
                scaled_disp, _ = self.scale_depth_disp(pred_resize)
                disp_img = self.generate_disparity_img(scaled_disp)
            else:
                disp_img = self.generate_disparity_img(1./pred_resize)

            imgname = os.path.basename(fname).split(".")[0]
            name_img = os.path.join(save_dir, imgname + ".jpeg")
            img.save(name_img)
            name_disp = os.path.join(save_dir, imgname+"_disp.jpeg")
            disp_img.save(name_disp)

    def scale_depth_disp(self, pred):  # TODO
        """
        Scale predicted depth and disparity from monodepth2
        """
        disp = 1. / pred
        min_disp = 1. / self.hparams.max_depth
        max_disp = 1. / self.hparams.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * ((disp - np.min(disp)) / (np.max(disp) - np.min(disp)))
        scaled_depth = 1. / scaled_disp
        return scaled_disp, scaled_depth

    def generate_disparity_img(self, disp):
        # generating colormapped disparity image
        vmax = np.percentile(disp, 95)
        normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        return im


def get_ckpt_path(dir_path):
    files = os.listdir(dir_path)
    files = [f for f in files if os.path.isfile(os.path.join(dir_path, f))]
    ckpt_path = None
    for f in files:
        if ".data" in f or ".meta" in f or ".index" in f or "checkpoint" in f or ".tune_metadata" in f:
            pass
        else:
            ckpt_path = os.path.join(dir_path, f)

    return ckpt_path


def main(args, logger):
    tester = ModelTester(args, log=logger)

    # read checkoints
    if os.path.isfile(args.ckpt_path):
        ckpts = [args.ckpt_path]
    else:
        ckpts = os.listdir(args.ckpt_path)
        ckpts.sort()
        ckpts = [os.path.join(args.ckpt_path, c) for c in ckpts]
        ckpts = [c for c in ckpts if os.path.isdir(c)]
        # get all checkpoint files
        ckpts = [get_ckpt_path(ckpt) for ckpt in ckpts]

    # iterate over checkpoints and evaluate model
    for ckpt in ckpts:
        logger.info("Evaluating for: {}".format(ckpt))
        tester.extract_model_spec(os.path.join(ckpt))
        preds = tester.run_evaluation(verbose=False)
        save_dir = os.path.join(os.path.dirname(ckpt), "predictions", args.name)

        if args.save_pred:
            logger.info("Saving predictions: {}".format(save_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            np.save(os.path.join(save_dir, "preds_all.npy"), preds)
            # save all predictions as images to disk
            tester.save_predictions(preds, save_dir, scale_pred=True)


def create_logger(args, save_dir, fname=None):
    """
    Create a logging object for logging
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    if fname is None:
        fname = 'stdout.log'
    hdlr = logging.FileHandler(os.path.join(save_dir, fname))
    hdlr.setLevel(logging.INFO)
    msg_format = '%(asctime)s [%(levelname)s] %(message)s'
    formatter = logging.Formatter(msg_format)
    ch.setFormatter(formatter)
    hdlr.setFormatter(formatter)
    root.addHandler(ch)
    root.addHandler(hdlr)
    logging.info(sys.version_info)
    logging.info(args)

    return logging


def create_parser():
    """Create arg parser for flags."""
    parser = argparse.ArgumentParser()
    # Dataset related flags
    parser.add_argument(
        '--image_path',
        required=True,
        help='directory or path to image'
    )
    parser.add_argument('--min_depth', type=float, default=1e-3, help="threshold for minimum depth for evaluation")
    parser.add_argument('--max_depth', type=float, default=80, help="threshold for maximum depth for evaluation")

    # Training settings
    parser.add_argument('--ckpt_path', type=str, required=True, help='If specified, tries to restore from given path.')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')
    parser.add_argument('--input_height', type=int, default=128, help='height of input image to model')
    parser.add_argument('--input_width', type=int, default=416, help='width of input image to model')
    parser.add_argument(
        '--dispnet_encoder',
        default='resnet50',
        choices=('resnet50', 'vgg'),
        help='type of encoder for dispnet'
    )
    parser.add_argument('--scale_normalize', action='store_true', help='spatially normalize depth prediction')
    parser.add_argument('--save_pred', action='store_true', help='save predictions on disk')
    parser.add_argument('--num_source', type=int, default=2, help='number of source images')
    parser.add_argument('--name', type=str, default='weather_internet')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_parser()
    log_dir = os.path.dirname(args.ckpt_path) if os.path.isfile(args.ckpt_path) else args.ckpt_path
    logger = create_logger(args, log_dir, fname=args.name + ".log")
    main(args, logger)

