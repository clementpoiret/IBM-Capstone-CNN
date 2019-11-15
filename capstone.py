# This program is developed to obtain the Advanced Data Science Certification
# offered by IBM and Coursera.
# Copyright (C) 2019 Clément POIRET
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Import Libraries
import os
import subprocess
import datetime
import numpy as np
import tensorflow as tf

import src.etl as etl
import src.model as md

# Global Variables
IMG_PATH = "/mnt/HDD/Documents/Datasets/AAF/faces/"


def get_gpus_memory():
    """Get the max gpu memory.

    Returns
    -------
    usage: list
        Returns a list of total memory for each gpus.
    """
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.total",
        "--format=csv,nounits,noheader"
    ]).decode("utf-8")

    gpus_memory = [int(x) for x in result.strip().split('\n')]
    return gpus_memory


def setup_gpus(allow_growth=True, memory_fraction=.9):
    """Setup GPUs.
    
    Parameters:
    allow_growth (Boolean)
    memory_fraction (Float): Set maximum memory usage, with 1 using
        maximum memory
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for i, gpu in enumerate(gpus):
                memory = get_gpus_memory()[i]

                tf.config.experimental.set_memory_growth(gpu, allow_growth)

                # Setting memory limit to max*fraction
                tf.config.experimental.set_virtual_device_configuration(
                    gpu, [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory * memory_fraction)
                    ])

                logical_gpus = tf.config.experimental.list_logical_devices(
                    "GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus),
                      "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def main():
    setup_gpus(allow_growth=True, memory_fraction=.95)

    if not os.path.exists("./data/train"):
        etl.arrange_images(path_in=IMG_PATH,
                           path_train="./data/train/",
                           path_test="./data/test/",
                           a=0,
                           b=7381,
                           name="female",
                           test_prop=.1,
                           seed=2019)

        etl.arrange_images(path_in=IMG_PATH,
                           path_train="./data/train/",
                           path_test="./data/test/",
                           a=7381,
                           b=13322,
                           name="male",
                           test_prop=.1,
                           seed=2019)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    classifier = md.build_classifier()

    training_set, test_set = etl.get_sets(
        "/home/clementpoiret/Documents/Datasets/data/train",
        "/home/clementpoiret/Documents/Datasets/data/test")

    history = classifier.fit_generator(training_set,
                                       epochs=128,
                                       validation_data=test_set,
                                       use_multiprocessing=True,
                                       callbacks=[tensorboard_callback])


if __name__ == "__main__":
    main()
