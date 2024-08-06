import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class AE(Model):
    def __init__(self, encoder: tf.keras.Sequential, decoder: tf.keras.Sequential):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def set_trainable(self, trainable):
        self.encoder.trainable = trainable
        self.decoder.trainable = trainable
