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
import datetime
import os
import pathlib

import numpy as np
import tensorflow as tf

import src.etl as etl
import src.hardware as hw
import src.model as md

# Global Variables
IMG_PATH = "/mnt/HDD/Documents/Datasets/AAF/faces/"


def main():
    hw.setup_gpus(allow_growth=True, memory_fraction=None)

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

    target_size = (224, 224)
    input_shape = (224, 224, 3)
    color_mode = "rgb"
    batch_size = 16

    print("Getting training set...")
    X_train, y_train, y_train_age = etl.get_sets(train_path="./data/train",
                                                 target_size=target_size,
                                                 color_mode=color_mode)

    print("Getting test set...")
    X_test, y_test, y_test_age = etl.get_sets(train_path="./data/test",
                                              target_size=target_size,
                                              color_mode=color_mode)

    _, n_genders = y_train.shape
    _, n_ages = y_train_age.shape

    print("Got {} genders for {} ages".format(n_genders, n_ages))

    loss_funcs = {
        "genders": "categorical_crossentropy",
        "ages": "categorical_crossentropy"
    }
    loss_weights = {"genders": 0.2, "ages": 1.0}
    metrics = {"genders": "accuracy", "ages": "accuracy"}
    optimizer = "adam"
    y_trains = {"genders": y_train, "ages": y_train_age}
    y_tests = {"genders": y_test, "ages": y_test_age}

    model = md.build_classifier(input_shape=input_shape,
                                n_genders=n_genders,
                                n_ages=n_ages,
                                optimizer=optimizer,
                                loss_funcs=loss_funcs,
                                loss_weights=loss_weights,
                                metrics=metrics)

    history = model.fit(x=X_train,
                        y=y_trains,
                        shuffle=True,
                        validation_data=(X_test, y_tests),
                        callbacks=[tensorboard_callback],
                        batch_size=batch_size,
                        epochs=35)


if __name__ == "__main__":
    main()
