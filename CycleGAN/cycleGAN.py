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
    Dropout(.5),
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
    loss='mean_squared_error',
)

cycle_gan.right_inverter.compile(
    optimizer='adam',
    loss='mean_squared_error',
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

history = []

for _ in train:
    try:
        fs_loss = f_gan.train_generator()  # f stacked
        fd_loss, fd_acc = f_gan.train_discriminator()  # g discriminator

        gs_loss = g_gan.train_generator()
        gd_loss, gd_acc = g_gan.train_discriminator()

        leftinv_loss = cycle_gan.train_left_inverse()
        rightinv_loss = cycle_gan.train_right_inverse()

        if train.log:
            print(f'{fs_loss:1.5f} {gs_loss:1.5f} {leftinv_loss:1.5f} {rightinv_loss:1.5f}')

            g_sample = g_gan.generate_sample(5)

            history.append(g_sample)

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
        train._compile_stats()
        break


def plot_image(image_array):
    """Plots the given 2D Image array (shape=image_dims) as grayscale image"""
    plt.tight_layout()
    plt.imshow(image_array, cmap='Greys_r')
    plt.axis('off')


def plot_samples(history):
    """Saves every image array in the 2D array `history` as file under ./samples"""
    for epoch_i, samples in enumerate(history):
        for sample_i, sample in enumerate(samples):
            plot_image(sample)
            plt.savefig(f'samples/{epoch_i:03.0f}_{sample_i:02.0f}.png')


plot_samples(history)

fig, (ax_stacked_loss, ax_inv_loss, ax_acc) = plt.subplots(3, 1)
fig.set_size_inches(16, 8)

ax_stacked_loss.set_title('stacked loss')
ax_stacked_loss.plot(train.stats.fs_loss)
ax_stacked_loss.plot(train.stats.gs_loss)
ax_stacked_loss.legend(('f', 'g'))

ax_inv_loss.set_title('cycle loss')
ax_inv_loss.plot(train.stats.leftinv_loss)
ax_inv_loss.plot(train.stats.rightinv_loss)
ax_inv_loss.legend(('left', 'right'))

ax_acc.set_title('discriminator accuracy')
ax_acc.plot(train.stats.fd_acc)
ax_acc.plot(train.stats.gd_acc)
ax_acc.legend(('f', 'g'))

fig.savefig('stats.png')
