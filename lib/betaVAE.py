import tensorflow as tf
from tensorflow.keras import layers, losses, metrics
from tensorflow.keras.models import Model


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def _create_encoder_model(encoder: tf.keras.Sequential, latent_dim: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(1800, 1))
    enc = encoder(inputs)
    mean = layers.Dense(latent_dim, name='mean')(enc)
    log_var = layers.Dense(latent_dim, name='log_var')(enc)
    z = Sampling()([mean, log_var])

    return tf.keras.Model(inputs, (mean, log_var, z), name="Encoder")


def _create_decoder_model(decoder: tf.keras.Sequential, latent_dim: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(latent_dim,))
    seq = decoder(inputs)
    return tf.keras.Model(inputs, seq, name="Decoder")


def get_seq_vae():
    encoder = tf.keras.Sequential([
        layers.Input(shape=(1800,)),
        layers.Reshape((1800, 1)),
        layers.Conv1D(filters=18, kernel_size=75, activation='relu', strides=4, padding="same"),
        layers.Conv1D(filters=28, kernel_size=30, activation='relu', strides=4, padding="same"),
        layers.Flatten(),
        layers.Dense(140, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(60, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(10, activation='relu'),
    ])

    decoder = tf.keras.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(10, activation='linear'),
        layers.Dense(20, activation='relu'),
        layers.Dense(60, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(140, activation='relu'),
        layers.Dense(113 * 28, activation='relu'),
        layers.Reshape((113, 28)),
        layers.Conv1DTranspose(filters=28, kernel_size=30, activation='relu', strides=4, padding="same",
                               output_padding=2),
        layers.Conv1DTranspose(filters=18, kernel_size=75, activation='relu', strides=4, padding="same"),
        layers.Conv1DTranspose(filters=1, kernel_size=81, activation='linear', padding="same"),
        layers.Reshape((1800,)),
    ])

    return encoder, decoder


class VAE(Model):
    def __init__(self, encoder_seq: tf.keras.Sequential, decoder_seq: tf.keras.Sequential, latent_dim: int):
        super(VAE, self).__init__()

        self.encoder = _create_encoder_model(encoder_seq, latent_dim)
        self.decoder = _create_decoder_model(decoder_seq, latent_dim)

        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.kl_weight = tf.Variable(1.0, trainable=False)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        return reconstruction

    @tf.function
    def calculate_loss(self, data, training: bool):
        data = data[0]
        z_mean, z_log_var, z = self.encoder(data, training=training)
        reconstruction = self.decoder(z, training=training)
        # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(losses.mean_squared_error(data, reconstruction), axis=(1,)))
        # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1,)))
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction), axis=(1,)))
        # reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction))

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) * self.kl_weight
        total_loss = reconstruction_loss + kl_loss

        if training:
            self.kl_weight.assign(self.kl_weight * 0.999)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return total_loss

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss = self.calculate_loss(data, True)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": self.total_loss_tracker.result(),
            "mse": self.reconstruction_loss_tracker.result(),
            "kl": self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        self.calculate_loss(data, False)

        return {
            "loss": self.total_loss_tracker.result(),
            "mse": self.reconstruction_loss_tracker.result(),
            "kl": self.kl_loss_tracker.result(),
        }
