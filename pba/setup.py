"""Parse flags and set up hyperparameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import tensorflow as tf

from pba.augmentation_transforms_hp import NUM_HP_TRANSFORM


def create_parser(state):
    """Create arg parser for flags."""
    parser = argparse.ArgumentParser()
    # Dataset related flags
    parser.add_argument(
        '--kitti_root',
        required=True,
        help='directory where KITTI processed dataset is located.')
    parser.add_argument(
        '--kitti_raw',
        required=True,
        help='directory where raw KITTI dataset is located.'
    )
    parser.add_argument(
        '--train_file_path',
        required=True,
        help='.txt file containing list of training image sequences')
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
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of threads for data loading, set zero to disable multiprocessing'
    )
    parser.add_argument('--min_depth', type=float, default=1e-3, help="threshold for minimum depth for evaluation")
    parser.add_argument('--max_depth', type=float, default=80, help="threshold for maximum depth for evaluation")

    # Training settings
    parser.add_argument('--local_dir', type=str, default='/tmp/ray_results/',  help='Ray directory.')
    parser.add_argument('--restore', type=str, default=None, help='If specified, tries to restore from given path.')
    parser.add_argument('--checkpoint_freq', type=int, default=50, help='Checkpoint frequency.')
    parser.add_argument('--cpu', type=float, default=4, help='Allocated by Ray')
    parser.add_argument('--gpu', type=float, default=1, help='Allocated by Ray')
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs, to run training/PBA search for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--lr_decay', default=None, choices=('step', 'cosine'))
    parser.add_argument('--batch_size', type=int, default=4, help='train batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')
    parser.add_argument('--input_height', type=int, default=128, help='height of input image to model')
    parser.add_argument('--input_width', type=int, default=416, help='width of input image to model')
    parser.add_argument('--num_samples', type=int, default=1, help='number of Ray samples')
    parser.add_argument('--max_outputs', type=int, default=4, help='how many minibatch per images we want to save')
    parser.add_argument('--use_regularization', action='store_true', help='whether or not to use regularization term')
    parser.add_argument(
        '--dispnet_encoder',
        default='resnet50',
        choices=('resnet50', 'vgg'),
        help='type of encoder for dispnet'
    )
    parser.add_argument('--scale_normalize', action='store_true', help='spatially normalize depth prediction')
    parser.add_argument('--rigid_warp_weight', type=float, default=1.0, help='weight for warping by rigid flow')
    parser.add_argument('--disp_smooth_weight', type=float, default=0.5)
    parser.add_argument('--num_scales', type=int, default=4, help='number of scaling points')
    parser.add_argument('--num_source', type=int, default=2, help='number of source images')
    parser.add_argument(
        '--alpha_recon_image',
        type=float,
        default=0.85,
        help='alpha weight between SSIM and L1 in reconstruction loss'
    )
    parser.add_argument(
        '--grad_clipping',
        type=float,
        default=0.0,
        help='gradient clipping by global norm, set to 0 to disable'
    )
    parser.add_argument(
        '--log_iter',
        type=int,
        default=50,
        help="logs image dato to comet.ml cloud by this interval"
    )

    # Policy settings
    if state == 'train':
        # PBA hp policy settings
        parser.add_argument(
            '--use_hp_policy',
            action='store_true',
            help='otherwise use autoaug policy')
        parser.add_argument(
            '--hp_policy',
            type=str,
            default=None,
            help='either a comma separated list of values or a file')
        parser.add_argument(
            '--hp_policy_epochs',
            type=int,
            default=30,
            help='number of epochs/iterations policy trained for')
        parser.add_argument(
            '--no_aug_policy',
            action='store_true',
            help='no augmentation policy at all, use kitti aug if enabled'
        )
        parser.add_argument(
            '--use_kitti_aug',
            action='store_true',
            help='use augmentation strategy from SIGNet for KITTI rather than using any aug policy'
        )
        parser.add_argument(
            '--flatten',
            action='store_true',
            help='randomly select an aug policy from schedule')
        parser.add_argument('--name', type=str, default='pba kitti')

    elif state == 'search':
        parser.add_argument('--perturbation_interval', type=int, default=10)
        parser.add_argument('--name', type=str, default='autoaug_pbt')
    else:
        raise ValueError('unknown state')

    # autoaugment policy setting it will be used only if hp policy is disabled
    parser.add_argument(
        '--policy_dataset',
        type=str,
        default='cifar10',
        choices=('cifar10', 'svhn'),
        help='which augmentation policy to use in case of autoaugment'
    )

    args = parser.parse_args()
    tf.logging.info(str(args))
    return args


def create_hparams(state, FLAGS):  # pylint: disable=invalid-name
    """
    Creates hyperparameters to pass into Ray config.

    Different options depending on search or eval mode.

    Args:
        state: a string, 'train' or 'search'.
        FLAGS: parsed command line flags.

    Returns:
        tf.hparams object.
    """
    tf.logging.info('data path: {}'.format(FLAGS.kitti_root))
    hparams = tf.contrib.training.HParams(
        kitti_root=FLAGS.kitti_root,
        kitti_raw=FLAGS.kitti_raw,
        train_file_path=FLAGS.train_file_path,
        test_file_path=FLAGS.test_file_path,
        gt_path=FLAGS.gt_path,
        num_workers=FLAGS.num_workers,
        min_depth=FLAGS.min_depth,
        max_depth=FLAGS.max_depth,
        batch_size=FLAGS.batch_size,
        test_batch_size=FLAGS.test_batch_size,
        lr=FLAGS.lr,
        lr_decay=FLAGS.lr_decay,
        input_height=FLAGS.input_height,
        input_width=FLAGS.input_width,
        max_outputs=FLAGS.max_outputs,
        use_regularization=FLAGS.use_regularization,
        dispnet_encoder=FLAGS.dispnet_encoder,
        scale_normalize=FLAGS.scale_normalize,
        rigid_warp_weight=FLAGS.rigid_warp_weight,
        disp_smooth_weight=FLAGS.disp_smooth_weight,
        num_scales=FLAGS.num_scales,
        num_source=FLAGS.num_source,
        alpha_recon_image=FLAGS.alpha_recon_image,
        gradient_clipping_by_global_norm=FLAGS.grad_clipping,
        policy_dataset=FLAGS.policy_dataset,
        name=FLAGS.name,
        log_iter=FLAGS.log_iter)

    if state == 'train':
        hparams.add_hparam('no_aug_policy', FLAGS.no_aug_policy)
        hparams.add_hparam('use_hp_policy', FLAGS.use_hp_policy)
        hparams.add_hparam('use_kitti_aug', FLAGS.use_kitti_aug)
        if FLAGS.use_hp_policy:
            if FLAGS.hp_policy == 'random':
                tf.logging.info('RANDOM SEARCH')
                parsed_policy = []
                for i in range(NUM_HP_TRANSFORM * 4):
                    if i % 2 == 0:
                        parsed_policy.append(random.randint(0, 10))
                    else:
                        parsed_policy.append(random.randint(0, 9))
            elif FLAGS.hp_policy.endswith('.txt') or FLAGS.hp_policy.endswith('.p'):
                # will be loaded in in data_utils
                parsed_policy = FLAGS.hp_policy
            else:
                # parse input into a fixed augmentation policy
                parsed_policy = FLAGS.hp_policy.split(', ')
                parsed_policy = [int(p) for p in parsed_policy]
            hparams.add_hparam('hp_policy', parsed_policy)
            hparams.add_hparam('hp_policy_epochs', FLAGS.hp_policy_epochs)
            hparams.add_hparam('flatten', FLAGS.flatten)
    elif state == 'search':
        hparams.add_hparam('no_aug_policy', False)
        hparams.add_hparam('use_kitti_aug', False)
        hparams.add_hparam('use_hp_policy', True)
        # default start value of 0
        hparams.add_hparam('hp_policy', [0 for _ in range(4 * NUM_HP_TRANSFORM)])
    else:
        raise ValueError('unknown state')

    epochs = FLAGS.epochs
    hparams.add_hparam('num_epochs', epochs)
    tf.logging.info('epochs: {}, lr: {}, lr_decay: {}'.format(hparams.num_epochs, hparams.lr, hparams.lr_decay))

    return hparams
