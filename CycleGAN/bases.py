import numpy as np

from keras.layers import Input
from keras import Model


class GAN:
    def __init__(self, *, generator: Model, discriminator: Model,
                 input_space: np.array, output_space: np.array, d_compile_args: dict, 
                 s_compile_args: dict):
        self.generator = generator
        self.discriminator = discriminator
        self.input_space = input_space
        self.output_space = output_space

        self.input_shape = input_space.shape[1:]
        self.output_shape = output_space.shape[1:]

        self.discriminator.compile(**d_compile_args)
        self.stacked = self._build_stacked()
        self.stacked.compile(**s_compile_args)

    def _build_stacked(self):
        self.set_trainable(self.discriminator, False)

        stacked_input = Input(self.input_shape)
        h = self.generator(stacked_input)
        validity = self.discriminator(h)
        return Model(stacked_input, validity)

    def train_generator(self, batch_size: int=32):
        x = self.random_sample(self.input_space, batch_size)
        y = np.ones((x.shape[0], 1))
        return self.stacked.train_on_batch(x, y)

    def train_discriminator(self, batch_size: int=32):
        return self.train_discriminator_on_batch(
            x_fake=self.random_sample(self.input_space, batch_size),
            real_sample=self.random_sample(self.output_space, batch_size)
        )

    def train_discriminator_on_batch(self, x_fake: np.array, real_sample: np.array):
        y_fake = np.zeros((x_fake.shape[0], 1))
        y_real = np.ones((real_sample.shape[0], 1))
        fake_sample = self.generator.predict(x_fake)
        return (
            self.discriminator.train_on_batch(fake_sample, y_fake),
            self.discriminator.train_on_batch(real_sample, y_real)
        )

    def generate_sample(self, batch_size: int=5):
        x = GAN.random_sample(self.input_space, batch_size)
        return self.generator.predict(x)

    @staticmethod
    def set_trainable(model: Model, flag: bool):
        model.trainable = flag
        for layer in model.layers:
            layer.trainable = flag

    @staticmethod
    def random_sample(space: np.array, batch_size: int=32):
        random_indices = np.random.choice(space.shape[0], batch_size)
        return space.take(random_indices, axis=0)  # batch dimension


class CycleGAN:
    def __init__(self, gan_a: GAN, gan_b: GAN):
        self.gan_a, self.gan_b = gan_a, gan_b

        assert gan_a.input_space is gan_b.output_space \
            and gan_b.input_space is gan_a.output_space, 'GANs must transform between the same spaces'

        self.space_a = gan_a.input_space
        self.space_b = gan_b.input_space

        self.left_inverter = self._build_stacked_inverter(gan_a, gan_b)
        self.right_inverter = self._build_stacked_inverter(gan_b, gan_a)

    # !IMPORTANT
    #   compile left and right inverse after initialisation

    def _build_stacked_inverter(self, f: GAN, g: GAN):
        original = Input(f.input_shape)
        translated = f.generator(original)
        reconstruction = g.generator(translated)
        return Model(original, reconstruction)

    def _train_stacked_inverter(self, inverter: Model, x_space: np.array, batch_size: int=32):
        x = GAN.random_sample(x_space, batch_size)
        return inverter.train_on_batch(x, x)

    def train_left_inverse(self, batch_size: int=32):
        return self._train_stacked_inverter(self.left_inverter, self.space_a, batch_size)

    def train_right_inverse(self, batch_size: int=32):
        return self._train_stacked_inverter(self.right_inverter, self.space_b, batch_size)
