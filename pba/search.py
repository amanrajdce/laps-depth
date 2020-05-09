"""Run PBA Search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import ray
from ray.tune import run_experiments
from ray.tune.schedulers import PopulationBasedTraining
import tensorflow as tf

from pba.setup import create_hparams
from pba.setup import create_parser
from pba.train import RayModel


def main(_):
    FLAGS = create_parser("search")  # pylint: disable=invalid-name
    hparams = create_hparams("search", FLAGS)
    hparams_config = hparams.values()

    train_spec = {
        "run": RayModel,
        "resources_per_trial": {
            "cpu": FLAGS.cpu,
            "gpu": FLAGS.gpu
        },
        "stop": {
            "training_iteration": hparams.num_epochs,
        },
        "config": hparams_config,
        "local_dir": FLAGS.local_dir,
        "checkpoint_freq": FLAGS.checkpoint_freq,
        "num_samples": FLAGS.num_samples
    }

    if FLAGS.restore:
        train_spec["restore"] = FLAGS.restore

    def explore(config):
        """
        Custom explore function based on cifar10 from PBA paper

        Args:
            config: dictionary containing ray config params.

        Returns:
        Copy of config with modified augmentation policy.
        """
        new_params = []

        for i, param in enumerate(config["hp_policy"]):
            if random.random() < 0.2:
                if i % 2 == 0:
                    new_params.append(random.randint(0, 10))
                else:
                    new_params.append(random.randint(0, 9))
            else:
                amt = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
                # Cast np.int64 to int for py3 json
                amt = int(amt)
                if random.random() < 0.5:
                    new_params.append(max(0, param - amt))
                else:
                    if i % 2 == 0:
                        new_params.append(min(10, param + amt))
                    else:
                        new_params.append(min(9, param + amt))
        config["hp_policy"] = new_params
        return config

    ray.init(
        webui_host='127.0.0.1',
        # memory=1024 * 1024 * 1024 * 20,    # setting 20 GB for ray workers
        # object_store_memory=1024 * 1024 * 1024 * 30,
        # lru_evict=True
    )

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr='abs_rel_acc',
        # metric="abs_rel",
        # mode="min",
        perturbation_interval=FLAGS.perturbation_interval,
        custom_explore_fn=explore,
        quantile_fraction=0.25,
        log_config=True
    )

    run_experiments(
        {
            FLAGS.name: train_spec
        },
        scheduler=pbt,
        reuse_actors=True,
        verbose=True)

    ray.shutdown()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
