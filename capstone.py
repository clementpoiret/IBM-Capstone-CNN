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

import numpy as np

import src.etl.etl as etl

# Global Variables
IMG_PATH = "/mnt/HDD/Documents/Datasets/AAF/faces/"


def main():
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

    train_generator = train_datagen.flow_from_directory(
        directory=r"./train/",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42)

    images = np.array(os.listdir(IMG_PATH))
    images.sort()

    ages = np.array([int(img[-6:-4]) for img in images])

    sns.distplot(ages)
    plt.show()
    ages.min()
    ages.max()
    ages.mean()
    ss.skew(ages)
    ss.kurtosis(ages)

    sns.distplot(ages[:7381])
    sns.distplot(ages[7381:])
    plt.savefig("dist.png")


if __name__ == "__main__":
    main()
