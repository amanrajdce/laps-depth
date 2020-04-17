# Mostly based on the code written by Clement Godard: 
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluate_kitti.py
from __future__ import division
import sys
import cv2
import os
import numpy as np
import time


def compute_errors(gt, pred):
    if np.size(gt)==0:
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


def setup_evaluation(gt_path, gt_interp_path, use_interp=False):
    """
    Read ground truth depth information
    :param gt_path: /path/to/gt_depth.npy
    :param gt_interp_path: /path/to/gt_interp_depth.npy
    :param use_interp: If enabled uses interpolated dense gt depth for evaluation
    :return: gt depth data
    """
    loaded_gt_depths = np.load(gt_path)
    loaded_gt_interps = np.load(gt_interp_path)

    num_test = len(loaded_gt_depths)
    gt_depths = []
    gt_interps = []

    for t_id in range(num_test):
        depth = loaded_gt_depths[t_id]
        interp = loaded_gt_interps[t_id]
        gt_depths.append(depth.astype(np.float32))
        gt_interps.append(interp.astype(np.float32))

    return gt_interps if use_interp else gt_depths


def run_evaluation(gt_depths, pred_depths, min_depth=1e-3, max_depth=80, verbose=False):
    t1 = time.time()
    num_test = len(gt_depths)
    pred_depths_resized = []

    for t_id in range(num_test):
        img_size_h, img_size_w = gt_depths[t_id].shape
        pred_depths_resized.append(cv2.resize(pred_depths[t_id], (img_size_w, img_size_h), interpolation=cv2.INTER_LINEAR))

    pred_depths = pred_depths_resized

    rms     = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)
                
    for i in range(num_test):    
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])
            
        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
        # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        gt_height, gt_width = gt_depth.shape

        crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
                        0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        # Scale matching
        scalar = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
        pred_depth[mask] *= scalar

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    if verbose:
        print('Evaluating took {:.4f} secs'.format(time.time()-t1))  # adapt to py3
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))

    return abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()
