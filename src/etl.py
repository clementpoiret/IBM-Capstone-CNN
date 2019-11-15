import os
import random

import numpy as np
import shutil
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


def get_sets(train_path, test_path):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=20,
                                       zoom_range=0.15,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.15,
                                       horizontal_flip=True,
                                       fill_mode="nearest")

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(256, 256),
                                                     color_mode="rgb",
                                                     batch_size=16,
                                                     class_mode="binary",
                                                     shuffle=True)

    test_set = test_datagen.flow_from_directory(test_path,
                                                target_size=(256, 256),
                                                batch_size=16,
                                                class_mode="binary")

    return training_set, test_set
