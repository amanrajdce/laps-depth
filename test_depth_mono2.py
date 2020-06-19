"""
Evaluates on KITTI eigen test split

python test_depth_mono2.py --pred_path /media/ehdd_2t/amanraj/monodepth2/results/kitti_raw_eigen_test.npy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import cv2
import tensorflow as tf
import numpy as np
import PIL.Image as pil
import pba.helper_utils as helper_utils

import matplotlib as mpl
import matplotlib.cm as cm
# mpl.use('Agg')
# import matplotlib.pyplot as plt


def read_test_files(dataset_dir, test_file_path):
    """
    Read kitti eigen test files
    :param dataset_dir: root path of kitti dataset
    :return: read test files with kitti root path added as prefix
    """
    with open(test_file_path, 'r') as f:
        test_files = f.readlines()
        test_files = [t.rstrip() for t in test_files]
        test_files = [os.path.join(dataset_dir, t) for t in test_files]

    return test_files


class Tester(object):
    def __init__(self, args, name, comet_exp=None):
        self.args = args
        self.gt_depths = np.load(args.gt_path)
        self.preds_all = np.load(args.pred_path)
        self.comet_exp = comet_exp
        self.test_file_path = "/ceph/amanraj/data/kitti_raw_eigen_test/test_files_eigen.txt"
        self.dataset_dir = "/ceph/amanraj/data/" + name

        self.test_files = read_test_files(self.dataset_dir, self.test_file_path)

    def run_evaluation(self, global_step=0, verbose=True):
        """Evaluate the child model.

        Args:
          global_step: step number at which to evaluate
          verbose: whether to print results

        Returns:
          results dictionary with error and accuracy metrics
        """

        # evaluate on predictions
        results = helper_utils.eval_predictions(
            self.gt_depths, self.preds_all, global_step, min_depth=self.args.min_depth,
            max_depth=self.args.max_depth, verbose=verbose, comet_exp=self.comet_exp
        )
        return results, self.preds_all

    def save_predictions(self, preds_all, save_dir, scale_pred=False):
        """
        Save predictions and image on disk
        :param preds_all: predicts from model
        :param scale_pred: if enabled scale predictions to min and max depth
        :return:
        """
        for idx, fname in enumerate(self.test_files):
            fh = open(fname, 'rb')
            img = pil.open(fh)
            orig_h, orig_w = self.gt_depths[idx].shape
            pred_resize = cv2.resize(preds_all[idx], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            if scale_pred:
                scaled_disp, _ = self.scale_depth_disp(pred_resize)
                disp_img = self.generate_disparity_img(scaled_disp)
            else:
                disp_img = self.generate_disparity_img(1./pred_resize)

            imgname = "{0:04d}".format(idx)
            name_img = os.path.join(save_dir, imgname+".jpeg")
            img.save(name_img)
            name_disp = os.path.join(save_dir, imgname+"_disp.jpeg")
            disp_img.save(name_disp)

    def scale_depth_disp(self, pred):  # TODO
        """
        Scale predicted depth and disparity from monodepth2
        """
        disp = 1. / pred
        min_disp = 1. / self.args.max_depth
        max_disp = 1. / self.args.min_depth
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


def main(args):
    fname = os.path.basename(args.pred_path).rstrip(".npy")
    tester = Tester(args, fname)
    results, preds = tester.run_evaluation(verbose=True)
    res = ",".join([str((k, v)) for k, v in results.items()])
    print(res)
    save_dir = os.path.join(args.save_dir, "predictions", fname)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save all predictions as images to disk
    tester.save_predictions(preds, save_dir)


def create_parser():
    """Create arg parser for flags."""
    parser = argparse.ArgumentParser()
    # Dataset related flags
    parser.add_argument(
        '--pred_path',
        required=True,
        help='.npy file containing predictions for kitti eigen test files'
    )
    parser.add_argument(
        '--gt_path',
        default='/ceph/amanraj/data/kitti_eigen_gt/gt_depth.npy',
        help='.npy file containing ground truth depth for kitti eigen test files'
    )
    parser.add_argument(
        '--save_dir',
        default='/media/ehdd_2t/amanraj/monodepth2/results',
        help='directory to save the generated results'
    )
    parser.add_argument('--min_depth', type=float, default=1e-3, help="threshold for minimum depth for evaluation")
    parser.add_argument('--max_depth', type=float, default=80, help="threshold for maximum depth for evaluation")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_parser()
    main(args)

