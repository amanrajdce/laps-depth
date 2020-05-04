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
from pba.model import ModelTrainer
from pba.sig_model import Model

import tensorflow as tf


class ModelTester(ModelTrainer):
    def __init__(self, hparams, comet_exp=None):
        self.hparams = hparams
        # Loading gt data and files for test set
        self.test_files = self.read_test_files(self.hparams.kitti_raw)
        self.gt_depths = self.setup_evaluation(self.hparams.gt_path)
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


def valid_ckpt_dir(dir_path, dir):
    """
    :param dir_path: absolute path of directory containing ckpt directory
    :param dir: name of current ckpt directory
    :return: returns True if checkpoint is available in the current directory
    """
    if "checkpoint" in dir:
        files = os.listdir(os.path.join(dir_path, dir))
        dir_fgp = ",".join(files)
        return True if ".data-" in dir_fgp else False
    else:
        return False


def main(args, logger):
    tester = ModelTester(args)
    ckpts = os.listdir(args.ckpt_dir)
    ckpts.sort()
    # get all checkpoint dirs having checkpoint data files
    ckpts = [os.path.join(args.ckpt_dir, d) for d in ckpts if valid_ckpt_dir(args.ckpt_dir, d)]

    # iterate over checkpoints and evaluate model
    for ckpt in ckpts:
        ckpt_num = ckpt.split("_")[-1]
        ckpt_path = os.path.join(ckpt, "model.ckpt-" + ckpt_num)
        logger.info("Evaluating for: {}".format(ckpt_path))
        tester.extract_model_spec(os.path.join(ckpt_path))
        preds_all = tester.eval_child_model(tester.meval, epoch=0)
        results = tester.run_evaluation(preds_all, epoch=0, verbose=False)
        res = ",".join([str((k, v)) for k, v in results.items()])
        logger.info("Result:{}".format(res))


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
        '--kitti_raw',
        required=True,
        help='directory where raw KITTI dataset is located.'
    )
    parser.add_argument(
        '--test_file_path',
        required=True,
        help='.txt file containing list of kitti eigen test files'
    )
    parser.add_argument(
        '--gt_path',
        required=True,
        help='.npy file containing ground truth depth for kitti eigen test files'
    )
    parser.add_argument('--min_depth', type=float, default=1e-3, help="threshold for minimum depth for evaluation")
    parser.add_argument('--max_depth', type=float, default=80, help="threshold for maximum depth for evaluation")

    # Training settings
    parser.add_argument('--ckpt_dir', type=str, default=None, help='If specified, tries to restore from given path.')
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
    parser.add_argument('--num_source', type=int, default=2, help='number of source images')
    parser.add_argument('--name', type=str, default='validation')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_parser()
    logger = create_logger(args, args.ckpt_dir, fname=args.name + ".log")
    main(args, logger)

