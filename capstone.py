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
import pathlib
import tensorflow as tf

import src.etl as etl
import src.model as md
import src.hardware as hw

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
    batch_size = 8

    X_train, y_train, y_train_age = etl.get_sets(train_path="./data/train",
                                                 test_path="./data/test",
                                                 target_size=target_size,
                                                 color_mode=color_mode)
    y_train_age = tf.keras.utils.to_categorical(y_train_age)
    #X_train = np.array([X_train[0]])
    #y_train_age = np.array([0, 1]).reshape(-1, 1)

    _, n_genders = y_train.shape
    _, n_ages = y_train_age.shape

    loss_funcs = {
        "genders": "categorical_crossentropy",
        "ages": "categorical_crossentropy"
    }
    loss_weights = {"genders": 0.8, "ages": 1.0}
    metrics = {"genders": "accuracy", "ages": "accuracy"}
    y_trains = {"genders": y_train, "ages": y_train_age}
    #    y_valids = {
    #        "genders": y_test,
    #        "ages": y_test_age
    #    }
    model = md.build_classifier(input_shape=input_shape,
                                n_genders=n_genders,
                                n_ages=n_ages,
                                loss_funcs=loss_funcs,
                                loss_weights=loss_weights,
                                metrics=metrics)

    history = model.fit(x=X_train,
                        y=y_trains,
                        shuffle=True,
                        callbacks=[tensorboard_callback],
                        batch_size=batch_size,
                        epochs=60)


if __name__ == "__main__":
    main()
