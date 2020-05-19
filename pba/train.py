"""Train and evaluate models using augmentation schedules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from comet_ml import Experiment
from comet_ml import OfflineExperiment

import ray
import os
from ray.tune import run_experiments
from ray.tune import Trainable
import shutil
import tensorflow as tf

from pba.model import ModelTrainer
from pba.setup import create_hparams
from pba.setup import create_parser


class RayModel(Trainable):
    """A Ray wrapper for Models to run search."""

    def _setup(self, *args):
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.logging.info("calling setup")
        self.hparams = tf.contrib.training.HParams(**self.config)
        self.hparams.add_hparam('trial_name', self.trial_name)

        if self.hparams.disable_comet:
            tf.logging.info("Started logging offline for comet ml")
            self.comet_experiment = OfflineExperiment(
                project_name=self.hparams.name, workspace="amanraj42",
                offline_directory=os.path.join(self.hparams.local_dir, self.hparams.name)
            )
        else:
            tf.logging.info("Started logging to comet ml online")
            self.comet_experiment = Experiment(
                api_key="1M6vRcdzjgqClytApluyGXhTP",
                project_name=self.hparams.name, workspace="amanraj42"
            )

        self.trainer = ModelTrainer(self.hparams, comet_exp=self.comet_experiment)

    # TODO, fix training, integrate KITTI evaluation
    def _train(self):
        """Runs one epoch(0 based, make it 1 based) of training, and returns current epoch errors."""
        tf.logging.info("training for iteration: {}".format(self._iteration + 1))
        results = self.trainer.run_model(self._iteration + 1)
        return results

    def _save(self, checkpoint_dir):
        """Uses tf trainer object to checkpoint. (1 based)"""
        save_name = self.trainer.save_model(checkpoint_dir, self._iteration)
        tf.logging.info("saved model {}".format(save_name))
        os.close(os.open(save_name, os.O_CREAT))
        return save_name

    def _restore(self, checkpoint):
        """Restores model from checkpoint."""
        tf.logging.info("RESTORING: {}".format(checkpoint))
        self.trainer.extract_model_spec(checkpoint)

    def reset_config(self, new_config):
        """Resets trainer config for fast PBT implementation."""
        self.config = new_config
        self.hparams = tf.contrib.training.HParams(**new_config)
        self.trainer.reset_config(self.hparams)
        return True


def main(_):
    FLAGS = create_parser("train")  # pylint: disable=invalid-name # TODO
    hparams = create_hparams("train", FLAGS)  # TODO

    train_spec = {
        "run": RayModel,
        "resources_per_trial": {
            "cpu": FLAGS.cpu,
            "gpu": FLAGS.gpu
        },
        "stop": {
            "training_iteration": hparams.num_epochs,
        },
        "config": hparams.values(),
        "local_dir": FLAGS.local_dir,
        "checkpoint_freq": FLAGS.checkpoint_freq,
        "num_samples": FLAGS.num_samples
    }

    if FLAGS.restore:
        train_spec["restore"] = FLAGS.restore

    ray.init(
        webui_host='127.0.0.1',
        # plasma_directory="/dev/shm/",
        # memory=1024 * 1024 * 1024 * 25,  # setting 25 GB for ray workers
        # object_store_memory=1024 * 1024 * 1024 * 5,  # setting 5 GB object store
        # lru_evict=True
    )

    # copy code to local_dir
    code_dir = os.path.join(os.path.abspath(os.getcwd()), 'pba')
    dst_dir = os.path.join(FLAGS.local_dir, FLAGS.name, 'pba')
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)  # remove old copy of code

    shutil.copytree(code_dir, dst_dir)

    run_experiments({FLAGS.name: train_spec})

    ray.shutdown()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
