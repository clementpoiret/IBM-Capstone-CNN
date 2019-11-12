import tensorflow as tf

from tensorflow.keras import layers, models


def build_model():
    model = models.Sequential()
    model.add(
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())

    model.add(layers.Dense(units=128, activation="relu"))

    # Output
    model.add(layers.Dense(units=1, activation="sigmoid"))

    # Compiling the CNN
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model
