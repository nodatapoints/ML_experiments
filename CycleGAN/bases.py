from keras.layers import Input

class GAN:
    def __init__(self, generator, discriminator,
                 g_compile_args={'optimizer':'adam', 'loss': 'binary_crossentropy'},
                 d_compile_args={'optimizer':'adam', 'loss': 'binary_crossentropy'}):
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator.compile(**d_compile_args)
        self.freeze(discriminator)

        stacked_input = Input(generator.input_shape)
        h = generator(stacked_input)
        validity = discriminator(h)
        self.stacked = Model(stacked_input, validity)
        self.stacked.compile(**g_compile_args)

    def train_generator(self, x_space, batch_size=32):
        x = self.random_sample(x_space, batch_size)
        y = np.ones((x.shape[0], 1))
        return self.stacked.train_on_batch(noise, real)

    def train_discriminator_on_batch(self, x_fake, real_sample):
        y_fake = np.zeros((x_fake.shape[0], 1))
        y_real = np.ones((real_sample.shape[0], 1))
        fake_sample = self.generator.predict(x_fake)
        return (
            self.discriminator.train_on_batch(fake_sample, y_fake),
            self.discriminator.train_on_batch(real_sample, y_real)
        )

    @staticmethod
    def freeze(model):
        model.trainable = False
        for layer in model.layers:
            layer.trainable = False

    @staticmethod
    def random_sample(space, batch_size=32):
        random_indices = np.random.choice(space.shape[0], batch_size)
        return space.take(random_indices, axis=0)  # batch dimension


    def train_discriminator(x_space, real_space, batch_size=32):
        return self.train_discriminator_on_batch(
            x_fake=self.random_sample(x_space, batch_size),
            real_sample=self.random_sample(real_space, batch_size)
        )
