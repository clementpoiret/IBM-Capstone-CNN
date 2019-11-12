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
import datetime
import numpy as np
import tensorflow as tf

import src.etl as etl
import src.model as md

# Global Variables
IMG_PATH = "/mnt/HDD/Documents/Datasets/AAF/faces/"


def setup_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus),
                      "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def main():
    setup_gpus()
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 1.0
    #session = tf.compat.v1.Session(config=config)

    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)
    #tf.config.experimental.set_virtual_device_configuration(
    #    gpus[0],
    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7979)])

    #tf.config.gpu.set_per_process_memory_fraction(1)
    #tf.config.gpu.set_per_process_memory_growth(True)

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

    model = md.build_model()

    training_set, test_set = etl.get_sets(
        "/home/clementpoiret/Documents/Datasets/data/train",
        "/home/clementpoiret/Documents/Datasets/data/test")

    history = model.fit_generator(training_set,
                                  steps_per_epoch=12028,
                                  epochs=25,
                                  validation_data=test_set,
                                  validation_steps=1294,
                                  callbacks=[tensorboard_callback])


if __name__ == "__main__":
    main()
