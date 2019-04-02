#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt

import idx2numpy

from keras.models import Model, Sequential
from keras.layers import Input, \
    Dense, \
    Flatten, \
    Reshape, \
    BatchNormalization, \
    Dropout, \
    Conv2D, \
    MaxPool2D, \
    UpSampling2D
from keras.optimizers import Adam

from bases import GAN, CycleGAN

# from keras.layers.advanced_activations import LeakyReLU
# from keras.optimizers import Adam

# Number of random variables for generator input
noise_dims = 10,

image_dims = 20, 20


images_space = idx2numpy.convert_from_file('circles.idx')

# normalize images_space
images_space = images_space / 127 - 1

noise_space = np.random.normal(0, 1, image_dims[:1] + noise_dims)


# Build generator
f = Sequential([
    Dense(32, activation='relu', input_shape=noise_dims),
    BatchNormalization(),
    Dense(64 * 10 * 10, activation='relu'),
    BatchNormalization(),
    Reshape((10, 10, 64)),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    UpSampling2D((2, 2)),
    Conv2D(16, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(1, (3, 3), padding='same', activation='tanh'),
    Reshape(image_dims)
])

f_d = Sequential([
    Reshape(image_dims + (1,), input_shape=image_dims),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(.3),
    Dense(1, activation='sigmoid')
])

f_gan = GAN(
    generator=f,
    discriminator=f_d,
    input_space=noise_space,
    output_space=images_space
)

f_gan.stacked.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)
f_gan.stacked.summary()

f_gan.discriminator.compile(
    optimizer=Adam(lr=1e-4),
    loss='binary_crossentropy'
)

g = Sequential([
    Reshape(image_dims + (1,), input_shape=image_dims),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(10, activation='relu')
])

g_d = Sequential([
    Dense(10, activation='relu', input_shape=noise_dims),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

g_gan = GAN(
    generator=g,
    discriminator=g_d,
    input_space=images_space,
    output_space=noise_space
)

g_gan.stacked.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

g_gan.discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

cycle_gan = CycleGAN(f_gan, g_gan)

cycle_gan.left_inverter.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

cycle_gan.right_inverter.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

f_gan.train_generator()
f_gan.train_discriminator()

g_gan.train_generator()
g_gan.train_discriminator()

cycle_gan.train_left_inverse()
cycle_gan.train_right_inverse()
