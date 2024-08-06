import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

from lib.plot_drawer import plot_predictions
from lib.plot_learning import PlotLearning, PlotLearning2


def print_model(autoencoder):
    tf.keras.utils.plot_model(autoencoder.encoder, to_file='encoder.png', show_shapes=True, show_layer_names=False)
    tf.keras.utils.plot_model(autoencoder.decoder, to_file='decoder.png', show_shapes=True, show_layer_names=False)


def mse(x, y):
    return (np.square(x - y)).mean()


def train_and_evaluate(autoencoder: Model, train_data, test_data, epochs_n=200, batch_size=30, show_latent=False,
                       patience=200, monitor="val_loss"):
    autoencoder.build(input_shape=(None, 1800))

    autoencoder.compile(optimizer='adam', loss="mse")

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=False, monitor=monitor)
    checkpoint_callback = ModelCheckpoint(filepath='best_model.weights.h5', save_best_only=True, monitor=monitor,
                                          mode='min', save_weights_only=True)

    if isinstance(train_data, tf.data.Dataset):
        train = (train_data.cache().batch(batch_size),)
        train_np = np.array([y for x, y in train_data])
        test = test_data.cache().batch(batch_size)
        test_np = np.array([y for x, y in test_data])
    else:
        train = (train_data, train_data)
        train_np = train_data
        test = (test_data, train_data)
        test_np = test_data

    autoencoder.fit(
        *train,
        epochs=epochs_n,
        shuffle=True,
        validation_data=test,
        verbose=1,
        callbacks=[early_stopping, checkpoint_callback, PlotLearning(autoencoder, show_latent=show_latent)],
        batch_size=batch_size
    )

    autoencoder.load_weights('best_model.weights.h5')
    autoencoder.save_weights("weights.h5")

    decoded_values_test = autoencoder.predict(test_np)
    decoded_values_train = autoencoder.predict(train_np)

    print(" ----- TEST SET ----- ")
    plot_predictions(test_np[:15], decoded_values_test[:15])

    print("\n\n\n ----- TRAIN SET ----- ")
    plot_predictions(train_np[:15], decoded_values_train[:15])
    #
    # print("\n\n\nTEST SET MSE:", mse(np.squeeze(test_np), np.squeeze(decoded_values_test)))
    # print("TRAINING SET MSE:", mse(np.squeeze(train_np), np.squeeze(decoded_values_train)))
