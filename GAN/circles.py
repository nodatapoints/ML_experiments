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

# from keras.layers.advanced_activations import LeakyReLU
# from keras.optimizers import Adam

# Number of random variables for generator input
noise_dims = 10

image_dims = 20, 20


images = idx2numpy.convert_from_file('circles.idx')

# normalize images
images = images / 127 - 1


# Build generator
model = Sequential([
    Dense(32, activation='relu', input_shape=(noise_dims, )),
    BatchNormalization(),
    Dense(64*10*10, activation='relu'),
    BatchNormalization(),
    Reshape((10, 10, 64)),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    UpSampling2D((2, 2)),
    Conv2D(16, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(1, (3, 3), padding='same', activation='tanh'),
    (Reshape(image_dims))
])
print('Generator')
model.summary()

noise = Input(shape=(noise_dims, ))
image_out = model(noise)

generator = Model(noise, image_out)

# Build discriminator

model = Sequential([
    Reshape(image_dims+(1,), input_shape=image_dims),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
print('Discriminator')
model.summary()

image_in = Input(shape=image_dims)
validity = model(image_in)
discriminator = Model(image_in, validity)

discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Stacked model
noise_in = Input(shape=(noise_dims,))
image = generator(noise_in)

discriminator.trainable = False
for layer in discriminator.layers:
    layer.trainable = False

validity = discriminator(image)

stacked = Model(noise_in, validity)

stacked.compile(
    optimizer='Adam',
    loss='binary_crossentropy'
)


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


def train_generator(batch_size=32):
    noise = np.random.normal(0, 1, (batch_size, noise_dims))
    real = np.ones((batch_size, 1))

    return stacked.train_on_batch(noise, real)


def plot_stats(stats):
    stacked_loss, loss_fake, loss_real, acc_fake, acc_real = zip(*stats)

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1)
    ax_loss.plot(stacked_loss)
    ax_loss.plot(loss_real)
    ax_loss.plot(loss_fake)
    ax_acc.plot(acc_fake)
    ax_acc.plot(acc_real)

    ax_loss.set_ylabel('loss')
    ax_loss.legend(('stacked', 'd real', 'd fake'))
    ax_acc.set_ylabel('accuracy')
    ax_acc.legend(('fake', 'real'))

    fig.savefig('stats.png', dpi=100)


def train_discriminator(batch_size=32):
    fake = np.zeros((batch_size, 1))
    real = np.ones((batch_size, 1))

    noise = np.random.normal(0, 1, (batch_size, noise_dims))
    random_indices = np.random.choice(images.shape[0], batch_size)

    real_images = images[random_indices, :, :]
    fake_images = generator.predict(noise)

    loss_fake, acc_fake = discriminator.train_on_batch(fake_images, fake)
    loss_real, acc_real = discriminator.train_on_batch(real_images, real)

    return loss_fake, loss_real, acc_fake, acc_real


# misc container for stats over all epochs
stats = []

# list of lists containing `n_samples` of random images for every 100th epoch
history = []

n_samples = 5

for epoch in range(1500):
    try:
        stacked_loss = train_generator()
        loss_fake, loss_real, acc_fake, acc_real = train_discriminator()

        # stats for later evaluation
        stats.append((stacked_loss, loss_fake, loss_real, acc_fake, acc_real))

        # Take samples every 100 epochs
        if epoch % 100 == 0:
            print(f'epoch {epoch:10}    loss stacked: {stacked_loss:2.3f}    acc fake: {acc_fake:0.5f}    acc real: {acc_real:0.5f}')

            samples = generator.predict(
                np.random.normal(0, 1, (n_samples, noise_dims)))
            history.append(samples)

    except KeyboardInterrupt:
        print(f'Aborted. Total epochs: {epoch}')
        break

history.append(
    generator.predict(np.random.normal(0, 1, (100, noise_dims)))
)

plot_samples(history)
plot_stats(stats)
