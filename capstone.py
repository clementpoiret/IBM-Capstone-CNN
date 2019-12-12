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
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import (Callback, CSVLogger, EarlyStopping,
                                        ModelCheckpoint)
from tensorflow.keras.preprocessing import image

import src.etl as etl
import src.hardware as hw
import src.model as md

# Global Variables
IMG_PATH = "/mnt/HDD/Documents/Datasets/AAF/faces/"


class TimeHistory(Callback):

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def main(train=True, weights=None, test_image=None):
    hw.setup_gpus(allow_growth=True, memory_fraction=None)

    if not os.path.exists("./data/train"):
        etl.arrange_images(path_in=IMG_PATH,
                           path_train="./data/train/",
                           path_test="./data/test/",
                           a=0,
                           b=7381,
                           name="female",
                           test_prop=0,
                           seed=2019)

        etl.arrange_images(path_in=IMG_PATH,
                           path_train="./data/train/",
                           path_test="./data/test/",
                           a=7381,
                           b=13322,
                           name="male",
                           test_prop=0,
                           seed=2019)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=30)

    target_size = (224, 224)
    input_shape = (224, 224, 3)
    color_mode = "rgb"
    batch_size = 8
    epochs = 100

    print("Getting training set...")
    X_train, y_train, y_train_age = etl.get_sets(train_path="./data/train",
                                                 target_size=target_size,
                                                 color_mode=color_mode)

    #print("Getting test set...")
    #X_test, y_test, y_test_age = etl.get_sets(train_path="./data/test",
    #                                          target_size=target_size,
    #                                          color_mode=color_mode)

    _, n_genders = y_train.shape
    _, n_ages = y_train_age.shape

    print("Got {} genders for {} ages".format(n_genders,
                                              len(np.unique(y_train_age))))

    loss_funcs = {
        "genders": "categorical_crossentropy",
        "ages": "mean_squared_error"
    }
    loss_weights = {"genders": 0.4, "ages": 1.0}
    metrics = {"genders": "accuracy", "ages": "mse"}
    optimizer = "adam"
    stages = 4
    bifurcation_stage = 3

    kf = KFold(n_splits=10, shuffle=True)

    if train:
        for i, (train_index,
                test_index) in enumerate(kf.split(X_train, y_train)):
            time_callback = TimeHistory()
            csv_logger = CSVLogger(
                "./logs/callbacks/training_v2_{}_{}_split{}.log".format(
                    stages, bifurcation_stage, i))
            mc = ModelCheckpoint("best_model_v2_{}_{}.h5".format(
                stages, bifurcation_stage, i),
                                 monitor="val_loss",
                                 mode="min",
                                 verbose=1)

            X_train_split, X_test_split = X_train[train_index], X_train[
                test_index]
            y_train_gender_split, y_test_gender_split = y_train[
                train_index], y_train[test_index]
            y_train_age_split, y_age_gender_split = y_train_age[
                train_index], y_train_age[test_index]

            y_trains = {
                "genders": y_train_gender_split,
                "ages": y_train_age_split
            }
            y_tests = {
                "genders": y_test_gender_split,
                "ages": y_age_gender_split
            }

            model = md.build_model(version=2,
                                   input_shape=input_shape,
                                   n_genders=n_genders,
                                   n_ages=n_ages,
                                   optimizer=optimizer,
                                   loss_funcs=loss_funcs,
                                   loss_weights=loss_weights,
                                   metrics=metrics,
                                   stages=stages,
                                   strides=[1, 2, 2, 2],
                                   n_identities=[2, 3, 5, 2],
                                   bifurcation_stage=bifurcation_stage)
            print(model.summary())

            model.fit(x=X_train_split,
                      y=y_trains,
                      shuffle=True,
                      validation_data=(X_test_split, y_tests),
                      callbacks=[
                          tensorboard_callback, es, time_callback, csv_logger,
                          mc
                      ],
                      batch_size=batch_size,
                      epochs=epochs)
    else:
        model = md.build_model(version=2,
                               input_shape=input_shape,
                               n_genders=n_genders,
                               n_ages=n_ages,
                               optimizer=optimizer,
                               loss_funcs=loss_funcs,
                               loss_weights=loss_weights,
                               metrics=metrics,
                               stages=stages,
                               strides=[1, 2, 2, 2],
                               n_identities=[2, 3, 5, 2],
                               bifurcation_stage=bifurcation_stage)
        print(model.summary())
        model.load_weights(weights)

        img = image.img_to_array(
            image.load_img(test_image,
                           color_mode=color_mode,
                           target_size=target_size,
                           interpolation="nearest")) / 255
        img = img.reshape(-1, 224, 224, 3)
        prediction = model.predict(img)
        if np.argmax(prediction[0]) == 0:
            g = "female"
        else:
            g = "male"
        print("The photo is showing a {} years old {}.".format(
            int(prediction[1][0][0] * 100), g))


if __name__ == "__main__":
    train = input("Do you want to train the model? Yes/No?")
    train = train == "Yes"
    if not train:
        weights = input("Path to your model's weights:")
        test_image = input("Path to your test image:")
    else:
        weights = None
        test_image = None

    main(train, weights, test_image)
