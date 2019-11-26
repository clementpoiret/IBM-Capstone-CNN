import os
import pathlib
import random
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def arrange_images(path_in,
                   path_train,
                   path_test,
                   a,
                   b,
                   name,
                   test_prop=.2,
                   seed=2019):

    random.seed(seed)

    images = np.array(os.listdir(path_in))
    images.sort()

    if not os.path.exists(path_train + name):
        os.makedirs(path_train + name)
    if not os.path.exists(path_test + name):
        os.makedirs(path_test + name)

    for image in images:
        if random.random() < test_prop:
            path = path_test
        else:
            path = path_train

        if (int(image[:5]) >= a) and (int(image[:5]) < b):
            shutil.copy(path_in + image, path + name + "/" + image)


train_path = "./data/train/"
target_size = (256, 256)
color_mode = "rgb"


def get_sets(train_path, target_size=(256, 256), color_mode="rgb"):
    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

    train_path = pathlib.Path(train_path)

    X_train = []
    y_train = []
    y_train_age = []

    c = 0
    for path in train_path.glob("*"):
        print("Loading class {}...".format(c))
        images = list(path.glob("*"))

        for cat_img in images:
            img = image.img_to_array(
                image.load_img(cat_img,
                               color_mode=color_mode,
                               target_size=target_size,
                               interpolation="nearest")) / 255

            X_train.append(img)
            y_train.append(c)
            y_train_age.append(int(cat_img.stem[6:]))
        c += 1

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_train_age = np.array(y_train_age, dtype=float).reshape(-1, 1)
    y_train_age -= y_train_age.min()
    y_train_age /= y_train_age.max()
    #y_train_age = tf.keras.utils.to_categorical(y_train_age)

    return X_train, y_train, y_train_age
