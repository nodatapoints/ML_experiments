import numpy as np

from keras.layers import Input
from keras import Model
from contextlib import contextmanager


class GAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

        self._build_stacked()

    def _build_stacked(self):
        with self.frozen(self.discriminator):
            stacked_input = Input(self.generator.input_shape)
            h = self.generator(stacked_input)
            validity = self.discriminator(h)
            self.stacked = Model(stacked_input, validity)
    # IMPORTANT
    # after initializing GAN stacked and discriminator have to
    # be compiled
    # gan.stacked.compile(...)
    # gan.discriminator.compile(...)

    def train_generator(self, x_space, batch_size=32):
        x = self.random_sample(x_space, batch_size)
        y = np.ones((x.shape[0], 1))
        return self.stacked.train_on_batch(x, y)

    def train_discriminator_on_batch(self, x_fake, real_sample):
        y_fake = np.zeros((x_fake.shape[0], 1))
        y_real = np.ones((real_sample.shape[0], 1))
        fake_sample = self.generator.predict(x_fake)
        return (
            self.discriminator.train_on_batch(fake_sample, y_fake),
            self.discriminator.train_on_batch(real_sample, y_real)
        )

    def train_discriminator(self, x_space, real_space, batch_size=32):
        return self.train_discriminator_on_batch(
            x_fake=self.random_sample(x_space, batch_size),
            real_sample=self.random_sample(real_space, batch_size)
        )

    @staticmethod
    def set_frozen(model, flag):
        model.trainable = flag
        for layer in model.layers:
            layer.trainable = flag

    @staticmethod
    @contextmanager
    def frozen(model):
        GAN.set_frozen(model, True)
        yield
        GAN.set_frozen(model, False)

    def random_sample(space, batch_size=32):
        random_indices = np.random.choice(space.shape[0], batch_size)
        return space.take(random_indices, axis=0)  # batch dimension