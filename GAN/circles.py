#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

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
                         MaxPool2D
# from keras.layers.advanced_activations import LeakyReLU
# from keras.optimizers import Adam

# Number of random variables for generator input
noise_dims = 10

image_dims = 20, 20


images = idx2numpy.convert_from_file('circles.idx')

# normalize images
images = images / 127 - 1


# Build generator
model = Sequential()

model.add(Dense(32, activation='relu'))
model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization(momentum=.9))
model.add(Dropout(.5))
model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization(momentum=.9))
model.add(Dropout(.5))
# model.add(Dense(1024, activation='relu'))
# model.add(BatchNormalization(momentum=.9))
# model.add(Dropout(.4))
model.add(Dense(np.prod(image_dims), activation='tanh'))
model.add(Reshape(image_dims))

noise = Input(shape=(noise_dims, ))
image_out = model(noise)

generator = Model(noise, image_out)


# Build discriminator

model = Sequential()

# model.add(Reshape(image_dims+(1,), input_shape=image_dims))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(.4))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

image_in = Input(shape=image_dims)
validity = model(image_in)
discriminator = Model(image_in, validity)

discriminator.compile(
    optimizer='Adam',
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
    plt.imshow(image_array, cmap='Greys_r')
    plt.axis('off')


def plot_samples(history):
    """Saves every image array in the 2D array `history` as file under ./samples"""
    for epoch_i, samples in enumerate(history):
        for sample_i, sample in enumerate(samples):
            plot_image(sample)
            # plt.savefig(f'samples/{epoch_i:03.0f}_{sample_i:02.0f}.png')  # Python 3
            plt.savefig('samples/%03.0f_%02.0f.png' % (epoch_i, sample_i))


def train_generator(batch_size=32):
    noise = np.random.normal(0, 1, (batch_size, noise_dims))
    real = np.ones((batch_size, 1))

    return stacked.train_on_batch(noise, real)


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


def plot_stats(stats):
    loss, acc_fake, acc_real = zip(*stats)

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1)

    ax_loss.plot(loss)
    ax_acc.plot(acc_fake)
    ax_acc.plot(acc_real)

    ax_loss.legend(('loss',))
    ax_acc.legend(('fake', 'real'))

    fig.savefig('stats.png')

# misc container for stats over all epochs
stats = []

# list of lists containing `n_samples` of random images for every 100th epoch
history = []

n_samples = 1
epoch = 1

while True:
    try:
        stacked_loss = train_generator()
        _, _, acc_fake, acc_real = train_discriminator()

        # stats for later evaluation
        stats.append((stacked_loss, acc_fake, acc_real))

        # Take samples every 100 epochs
        if epoch % 100 == 0:
            # print(f'epoch {epoch:10}    loss stacked: {stacked_loss:2.3f}    acc fake: {acc_fake:0.5f}    acc real: {acc_real:0.5f}')  # Python 3
            print('epoch %04d    loss stacked: %2.3f    acc fake: %.5f    acc real: %.5f' % (epoch, stacked_loss, acc_fake, acc_real) )

            samples = generator.predict(
                np.random.normal(0, 1, (n_samples, noise_dims)))
            history.append(samples)

        epoch += 1

    except KeyboardInterrupt:
        print('Aborted. Total epochs:', epoch)
        break

plot_samples(history)
plot_stats(stats)
