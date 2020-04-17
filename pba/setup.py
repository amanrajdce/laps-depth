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
        help='Directory where KITTI processed dataset is located.')
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
    parser.add_argument('--local_dir', type=str, default='/tmp/ray_results/',  help='Ray directory.')
    parser.add_argument('--restore', type=str, default=None, help='If specified, tries to restore from given path.')
    parser.add_argument('--checkpoint_freq', type=int, default=50, help='Checkpoint frequency.')
    parser.add_argument(
        '--cpu', type=float, default=4, help='Allocated by Ray')
    parser.add_argument(
        '--gpu', type=float, default=1, help='Allocated by Ray')
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs, to run training/PBA search for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--lr_decay', default=None, choices=('step', 'cosine'))
    parser.add_argument('--batch_size', type=int, default=4, help='train batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of Ray samples')

    if state == 'train':
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
            '--no_aug',
            action='store_true',
            help=
            'no additional augmentation at all (besides cutout if not toggled)'
        )
        parser.add_argument(
            '--flatten',
            action='store_true',
            help='randomly select aug policy from schedule')
        parser.add_argument('--name', type=str, default='autoaug')

    elif state == 'search':
        parser.add_argument('--perturbation_interval', type=int, default=10)
        parser.add_argument('--name', type=str, default='autoaug_pbt')
    else:
        raise ValueError('unknown state')
    args = parser.parse_args()
    tf.logging.info(str(args))
    return args


def create_hparams(state, FLAGS):  # pylint: disable=invalid-name
    """Creates hyperparameters to pass into Ray config.

  Different options depending on search or eval mode.

  Args:
    state: a string, 'train' or 'search'.
    FLAGS: parsed command line flags.

  Returns:
    tf.hparams object.
  """
    epochs = 0
    tf.logging.info('data path: {}'.format(FLAGS.kitti_root))
    hparams = tf.contrib.training.HParams(
        kitti_root=FLAGS.kitti_root,
        batch_size=FLAGS.batch_size,
        test_batch_size=FLAGS.test_batch_size,
        gradient_clipping_by_global_norm=5.0,
        explore=FLAGS.explore,
        aug_policy=FLAGS.aug_policy,
        lr=FLAGS.lr)

    if state == 'train':
        hparams.add_hparam('no_aug', FLAGS.no_aug)
        hparams.add_hparam('use_hp_policy', FLAGS.use_hp_policy)
        if FLAGS.use_hp_policy:
            if FLAGS.hp_policy == 'random':
                tf.logging.info('RANDOM SEARCH')
                parsed_policy = []
                for i in range(NUM_HP_TRANSFORM * 4):
                    if i % 2 == 0:
                        parsed_policy.append(random.randint(0, 10))
                    else:
                        parsed_policy.append(random.randint(0, 9))
            elif FLAGS.hp_policy.endswith('.txt') or FLAGS.hp_policy.endswith(
                    '.p'):
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
        hparams.add_hparam('no_aug', False)
        hparams.add_hparam('use_hp_policy', True)
        # default start value of 0
        hparams.add_hparam('hp_policy', [0 for _ in range(4 * NUM_HP_TRANSFORM)])
    else:
        raise ValueError('unknown state')

    if FLAGS.epochs > 0:
        tf.logging.info('overwriting with custom epochs')
        epochs = FLAGS.epochs
    hparams.add_hparam('num_epochs', epochs)
    tf.logging.info('epochs: {}, lr: {}, lr_decay: {}'.format(hparams.num_epochs, hparams.lr, hparams.lr_decay))

    return hparams
