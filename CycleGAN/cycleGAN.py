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
from training import Training
from plots import plot_stats, plot_samples

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
    output_space=images_space,
    d_compile_args=dict(
        optimizer=Adam(lr=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    ),
    s_compile_args=dict(
        optimizer='adam',
        loss='binary_crossentropy'
    )
)

g = Sequential([
    Reshape(image_dims + (1,), input_shape=image_dims),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='tanh')
])

g_d = Sequential([
    Dense(256, activation='relu', input_shape=noise_dims),
    Dense(64, activation='relu'),
    Dropout(.5),
    Dense(64, activation='relu'),
    Dropout(.5),
    Dense(1, activation='sigmoid')
])

g_gan = GAN(
    generator=g,
    discriminator=g_d,
    input_space=images_space,
    output_space=noise_space,
    d_compile_args=dict(
        optimizer=Adam(lr=9e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    ),
    s_compile_args=dict(
        optimizer='adam',
        loss='binary_crossentropy'
    )
)

cycle_gan = CycleGAN(
    gan_a=f_gan,
    gan_b=g_gan,
    l_compile_args=dict(
        optimizer='adam',
        loss='mean_squared_error'
    ),
    r_compile_args=dict(
        optimizer='adam',
        loss='mean_squared_error'
    )
)

train = Training(
    n_generations=1000,
    log_period=100,
    stat_entries=(
        'fs_loss',
        'fd_loss',
        'fd_acc',
        'gs_loss',
        'gd_loss',
        'gd_acc',
        'leftinv_loss',
        'rightinv_loss',
    )
)

try:
    for _ in train:
        # fs_loss = f_gan.train_generator()  # f stacked
        # fd_loss, fd_acc = f_gan.train_discriminator()  # g discriminator

        gs_loss = g_gan.train_generator()
        gd_loss, gd_acc = g_gan.train_discriminator()

        # leftinv_loss = cycle_gan.train_left_inverse()
        # rightinv_loss = cycle_gan.train_right_inverse()

        fs_loss = 0
        fd_loss, fd_acc = 0, 0
        leftinv_loss, rightinv_loss = 0, 0

        if train.log:
            print(
                f'{fs_loss:1.5f} {gs_loss:1.5f} {fd_acc:.5f} {gd_acc:.5f}'
            )

        if train.make_samples:
            g_sample = g_gan.generate_sample(5).reshape((-1, 5, 2))
            train.append_sample(g_sample, 'samples/sample_{gen:03.0f}_{i:02.0f}.png')

        train.append_stats(
            fs_loss,
            fd_loss,
            fd_acc,
            gs_loss,
            gd_loss,
            gd_acc,
            leftinv_loss,
            rightinv_loss
        )

except KeyboardInterrupt:
    pass

else:
    train.compile_records()
    plot_samples(train.samples)
    plot_stats(train.stats)
