import numpy as np
from string import ascii_lowercase

import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Activation, Add, AveragePooling2D,
                                     BatchNormalization, Conv2D, Dense, Dropout,
                                     Flatten, Input, MaxPooling2D,
                                     ZeroPadding2D)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model


def identity_block_v1(X, f, filters, stage, block):
    print("identity_block_v1_{}_{}_{}_{}_{}".format(X, f, filters, stage,
                                                    block))
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


def identity_block_v2(X, f, filters, stage, block):
    print("identity_block_v2_{}_{}_{}_{}_{}".format(X, f, filters, stage,
                                                    block))
    conv_name_base = "res{}{}_branch".format(stage, block)
    bn_name_base = "bn{}{}_branch".format(stage, block)

    F1, F2, F3 = filters

    X_shortcut = X

    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)
    X = Conv2D(filters=F1,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding="valid",
               name=conv_name_base + "2a",
               kernel_initializer=GlorotUniform(seed=2019))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)
    X = Conv2D(filters=F2,
               kernel_size=(f, f),
               strides=(1, 1),
               padding="same",
               name=conv_name_base + "2b",
               kernel_initializer=GlorotUniform(seed=2019))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
    X = Activation("relu")(X)
    X = Conv2D(filters=F3,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding="valid",
               name=conv_name_base + "2c",
               kernel_initializer=GlorotUniform(seed=2019))(X)

    X = Add()([X, X_shortcut])

    return X


def convolutional_block_v1(X, f, filters, stage, block, s=2):
    print("convolutional_block_v1_{}_{}_{}_{}_{}_{}".format(
        X, f, filters, stage, block, s))
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


def convolutional_block_v2(X, f, filters, stage, block, s=2):
    print("convolutional_block_v2_{}_{}_{}_{}_{}_{}".format(
        X, f, filters, stage, block, s))
    conv_name_base = "res{}{}_branch".format(stage, block)
    bn_name_base = "bn{}{}_branch".format(stage, block)

    F1, F2, F3 = filters

    X_shortcut = X

    if s == 1:
        X = Conv2D(filters=F1,
                   kernel_size=(1, 1),
                   strides=(s, s),
                   padding="valid",
                   name=conv_name_base + "2a",
                   kernel_initializer=GlorotUniform(seed=2019))(X)
    else:
        X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
        X = Activation("relu")(X)
        X = Conv2D(filters=F1,
                   kernel_size=(1, 1),
                   strides=(s, s),
                   padding="valid",
                   name=conv_name_base + "2a",
                   kernel_initializer=GlorotUniform(seed=2019))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)
    X = Conv2D(filters=F2,
               kernel_size=(f, f),
               strides=(1, 1),
               padding="same",
               name=conv_name_base + "2b",
               kernel_initializer=GlorotUniform(seed=2019))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
    X = Activation("relu")(X)
    X = Conv2D(filters=F3,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding="valid",
               name=conv_name_base + "2c",
               kernel_initializer=GlorotUniform(seed=2019))(X)

    X_shortcut = Conv2D(filters=F3,
                        kernel_size=(1, 1),
                        strides=(s, s),
                        padding="valid",
                        name=conv_name_base + "1",
                        kernel_initializer=GlorotUniform(seed=2019))(X_shortcut)

    X = Add()([X, X_shortcut])

    return X


def build_model(version,
                loss_funcs,
                n_genders,
                n_ages,
                metrics,
                loss_weights,
                optimizer,
                input_shape=(256, 256, 3),
                stages=4,
                strides=[1, 2, 2, 2],
                n_identities=[2, 3, 5, 2],
                initial_filter=np.array([64, 64, 256]),
                bifurcation_stage=3):

    if version == 1:
        convolutional_block = convolutional_block_v1
        identity_block = identity_block_v1
    elif version == 2:
        convolutional_block = convolutional_block_v2
        identity_block = identity_block_v2
    else:
        raise NameError("Unsupported Version '{}'".format(version))

    X_input = Input(shape=input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7),
               strides=(2, 2),
               name="conv1",
               kernel_initializer=GlorotUniform(seed=2019))(X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    for stage in range(stages):
        s = strides[stage]

        if stage < bifurcation_stage:
            X = convolutional_block(X,
                                    f=3,
                                    filters=list(initial_filter * 2**stage),
                                    stage=stage + 2,
                                    block="a",
                                    s=s)

            n_identity = n_identities[stage]
            for block in ascii_lowercase[1:n_identity + 1]:
                X = identity_block(X,
                                   3,
                                   filters=list(initial_filter * 2**stage),
                                   stage=stage + 2,
                                   block=block)
        else:
            if stage == bifurcation_stage:
                genders = X
                ages = X
            genders = convolutional_block(genders,
                                          f=3,
                                          filters=list(initial_filter *
                                                       2**stage),
                                          stage=stage + 2,
                                          block="gender_a",
                                          s=s)
            ages = convolutional_block(ages,
                                       f=3,
                                       filters=list(initial_filter * 2**stage),
                                       stage=stage + 2,
                                       block="age_a",
                                       s=s)

            n_identity = n_identities[stage]
            for block in ascii_lowercase[1:n_identity + 1]:
                genders = identity_block(genders,
                                         3,
                                         filters=list(initial_filter *
                                                      2**stage),
                                         stage=stage + 2,
                                         block="gender_{}".format(block))
                ages = identity_block(ages,
                                      3,
                                      filters=list(initial_filter * 2**stage),
                                      stage=stage + 2,
                                      block="age_{}".format(block))

    if bifurcation_stage == stages:
        genders = X
        ages = X

    if version == 2:
        genders = BatchNormalization(axis=3)(genders)
        ages = BatchNormalization(axis=3)(ages)

        genders = Activation("relu")(genders)
        ages = Activation("relu")(ages)

    genders = AveragePooling2D(pool_size=(2, 2), padding="same")(genders)
    ages = AveragePooling2D(pool_size=(2, 2), padding="same")(ages)

    genders = Flatten()(genders)
    ages = Flatten()(ages)

    genders = Dense(256,
                    activation="relu",
                    name="fc_genders_256_1",
                    kernel_initializer=GlorotUniform(seed=2019),
                    kernel_regularizer=regularizers.l1_l2(0.0001,
                                                          0.0001))(genders)
    genders = Dropout(rate=.2, seed=2019)(genders)
    ages = Dense(256,
                 activation="relu",
                 name="fc_ages_256_1",
                 kernel_initializer=GlorotUniform(seed=2019),
                 kernel_regularizer=regularizers.l1_l2(0.0001, 0.0001))(ages)
    ages = Dropout(rate=.2, seed=2019)(ages)

    genders = Dense(128,
                    activation="relu",
                    name="fc_genders_256_2",
                    kernel_initializer=GlorotUniform(seed=2019),
                    kernel_regularizer=regularizers.l1_l2(0.0001,
                                                          0.0001))(genders)
    genders = Dropout(rate=.2, seed=2019)(genders)
    ages = Dense(128,
                 activation="relu",
                 name="fc_ages_256_2",
                 kernel_initializer=GlorotUniform(seed=2019),
                 kernel_regularizer=regularizers.l1_l2(0.0001, 0.0001))(ages)
    ages = Dropout(rate=.2, seed=2019)(ages)

    genders = Dense(n_genders,
                    activation="softmax",
                    name="genders",
                    kernel_initializer=GlorotUniform(seed=2019))(genders)
    ages = Dense(n_ages,
                 activation="sigmoid",
                 name="ages",
                 kernel_initializer=GlorotUniform(seed=2019))(ages)

    model = Model(inputs=X_input,
                  outputs=[genders, ages],
                  name="ResNet50_mod{}{}".format(stages, bifurcation_stage))

    model.compile(optimizer=optimizer,
                  loss=loss_funcs,
                  loss_weights=loss_weights,
                  metrics=metrics)

    plot_model(model,
               to_file="model_loop_{}_{}_v{}.png".format(
                   stages, bifurcation_stage, version))

    return model
