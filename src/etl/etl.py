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
