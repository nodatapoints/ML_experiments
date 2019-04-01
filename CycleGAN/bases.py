import numpy as np

from keras.layers import Input
from keras import Model
from contextlib import contextmanager


class GAN:
    def __init__(self, *, generator, discriminator, input_space, output_space):
        self.generator = generator
        self.discriminator = discriminator
        self.input_space = input_space
        self.output_space = output_space

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

    def train_generator(self, batch_size=32):
        x = self.random_sample(self.input_space, batch_size)
        y = np.ones((x.shape[0], 1))
        return self.stacked.train_on_batch(x, y)

    def train_discriminator(self, batch_size=32):
        return self.train_discriminator_on_batch(
            x_fake=self.random_sample(self.input_space, batch_size),
            real_sample=self.random_sample(self.output_space, batch_size)
        )

    def train_discriminator_on_batch(self, x_fake, real_sample):
        y_fake = np.zeros((x_fake.shape[0], 1))
        y_real = np.ones((real_sample.shape[0], 1))
        fake_sample = self.generator.predict(x_fake)
        return (
            self.discriminator.train_on_batch(fake_sample, y_fake),
            self.discriminator.train_on_batch(real_sample, y_real)
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


class CycleGAN:
    def __init__(self, *, gan_a, gan_b, space_a, space_b):
        self.gan_a, self.gan_b = gan_a, gan_b
        self.space_a, self.space_b = space_a, space_b

        self.left_inverter = self._build_stacked_inverter(gan_a, gan_b)
        self.right_inverter = self._build_stacked_inverter(gan_b, gan_a)

    # !IMPORTANT
    #   compile left and right inverse after initialisation

    def _build_stacked_inverter(self, f, g):
        original = Input(f.input_shape)
        translated = f(original)
        reconstruction = g(translated)
        return Model(original, reconstruction)

    def _train_stacked_inverter(self, inverter, x_space, batch_size=32):
        x = GAN.random_sample(x_space, batch_size)
        return inverter.train_on_batch(x, x)

    def train_left_inverse(self, batch_size=32):
        return self._train_stacked_inverter(self.left_inverter, self.space_a, batch_size)

    def train_right_inverse(self, batch_size=32):
        return self._train_stacked_inverter(self.right_inverter, self.space_b, batch_size)
