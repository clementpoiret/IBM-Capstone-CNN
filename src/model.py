import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Flatten, Activation


def build_classifier():
    classifier = Sequential()

    classifier.add(
        Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    classifier.add(Dropout(0.2))
    classifier.add(MaxPooling2D((2, 2)))

    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(MaxPooling2D((2, 2)))

    classifier.add(Conv2D(256, (3, 3), activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Conv2D(256, (3, 3), activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(MaxPooling2D((2, 2)))

    classifier.add(Conv2D(512, (3, 3), activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Conv2D(512, (3, 3), activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(MaxPooling2D((2, 2)))

    classifier.add(Conv2D(512, (3, 3), activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Conv2D(512, (3, 3), activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(MaxPooling2D((2, 2)))

    classifier.add(Flatten())

    classifier.add(Dense(units=2048, activation="relu"))

    classifier.add(Dense(units=1024, activation="relu"))

    classifier.add(Dense(units=256, activation="relu"))

    classifier.add(Dense(units=64, activation="relu"))

    classifier.add(Dense(units=1, activation="sigmoid"))

    classifier.compile(optimizer="adam",
                       loss="binary_crossentropy",
                       metrics=["accuracy"])

    return classifier
