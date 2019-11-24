import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                     BatchNormalization, Conv2D, Dense, Dropout,
                                     Flatten, Input, MaxPooling2D,
                                     ZeroPadding2D, Add)
from tensorflow.keras.models import Model


def identity_block(X, f, filters, stage, block):
    conv_name_base = "res{}{}_branch".format(stage, block)
    bn_name_base = "bn{}{}_branch".format(stage, block)

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding="valid",
               name=conv_name_base + "2a",
               kernel_initializer=GlorotUniform(seed=2019))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    X = Conv2D(filters=F2,
               kernel_size=(f, f),
               strides=(1, 1),
               padding="same",
               name=conv_name_base + "2b",
               kernel_initializer=GlorotUniform(seed=2019))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    X = Conv2D(filters=F3,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding="valid",
               name=conv_name_base + "2c",
               kernel_initializer=GlorotUniform(seed=2019))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
    X = Activation("relu")(X)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = "res{}{}_branch".format(stage, block)
    bn_name_base = "bn{}{}_branch".format(stage, block)

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1,
               kernel_size=(1, 1),
               strides=(s, s),
               padding="valid",
               name=conv_name_base + "2a",
               kernel_initializer=GlorotUniform(seed=2019))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    X = Conv2D(filters=F2,
               kernel_size=(f, f),
               strides=(1, 1),
               padding="same",
               name=conv_name_base + "2b",
               kernel_initializer=GlorotUniform(seed=2019))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    X = Conv2D(filters=F3,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding="valid",
               name=conv_name_base + "2c",
               kernel_initializer=GlorotUniform(seed=2019))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
    X = Activation("relu")(X)

    X_shortcut = Conv2D(filters=F3,
                        kernel_size=(1, 1),
                        strides=(s, s),
                        padding="valid",
                        name=conv_name_base + "1",
                        kernel_initializer=GlorotUniform(seed=2019))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + "1")(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def build_classifier(input_shape=(256, 256, 3), classes=2):
    X_input = Input(shape=input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7),
               strides=(2, 2),
               name="conv1",
               kernel_initializer=GlorotUniform(seed=2019))(X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X,
                            f=3,
                            filters=[64, 64, 256],
                            stage=2,
                            block="a",
                            s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block="b")
    X = identity_block(X, 3, [64, 64, 256], stage=2, block="c")

    X = convolutional_block(X,
                            f=3,
                            filters=[128, 128, 512],
                            stage=3,
                            block="a",
                            s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block="b")
    X = identity_block(X, 3, [128, 128, 512], stage=3, block="c")
    X = identity_block(X, 3, [128, 128, 512], stage=3, block="d")

    X = convolutional_block(X,
                            f=3,
                            filters=[256, 256, 1024],
                            stage=4,
                            block="a",
                            s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="b")
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="c")
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="d")
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="e")
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="f")

    X = convolutional_block(X,
                            f=3,
                            filters=[512, 512, 2048],
                            stage=5,
                            block="a",
                            s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block="b")
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block="c")

    X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)

    X = Flatten()(X)

    act_function = "softmax"
    loss = "categorical_crossentropy"
    n_outputs = classes

    X = Dense(n_outputs,
              activation=act_function,
              name="fc" + str(n_outputs),
              kernel_initializer=GlorotUniform(seed=2019))(X)

    classifier = Model(inputs=X_input, outputs=X, name="ResNet50")

    classifier.compile(optimizer="SGD", loss=loss, metrics=["accuracy"])

    return classifier
