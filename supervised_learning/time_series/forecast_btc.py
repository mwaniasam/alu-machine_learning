#!/usr/bin/env python3
"""
Creates, trains, and validates a Keras LSTM model
for BTC price forecasting using the past 24 hours
of data to predict the close price of the next hour.
"""
import numpy as np
import tensorflow as tf


def create_dataset(data, window_size=24):
    """
    Creates a tf.data.Dataset from time series data using
    a sliding window approach.

    Args:
        data: numpy.ndarray - normalized time series data
        window_size: int - number of past hours to use (default 24)

    Returns:
        tf.data.Dataset - dataset of (input_window, target) pairs
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)
    # window of size+1: size inputs + 1 target
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(
        lambda w: w.batch(window_size + 1))
    dataset = dataset.map(
        lambda w: (w[:-1], w[-1]))

    return dataset


def split_data(data, train_ratio=0.7, val_ratio=0.2):
    """
    Splits data into train, validation, and test sets.

    Args:
        data: numpy.ndarray - full dataset
        train_ratio: float - proportion for training
        val_ratio: float - proportion for validation

    Returns:
        tuple: (train_data, val_data, test_data)
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def build_model(window_size=24):
    """
    Builds and compiles the LSTM model for BTC forecasting.

    Args:
        window_size: int - number of time steps in input sequence

    Returns:
        tf.keras.Model - compiled LSTM model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            64,
            return_sequences=True,
            input_shape=(window_size, 1)
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def forecast():
    """
    Main function to load preprocessed data, build datasets,
    train the LSTM model, and evaluate it on validation data.
    """
    # Load preprocessed data
    print("Loading preprocessed data...")
    data = np.load('close_prices.npy').astype(np.float32)

    # Split data
    train_data, val_data, test_data = split_data(data)
    print("Train size: {}, Val size: {}, Test size: {}".format(
        len(train_data), len(val_data), len(test_data)))

    window_size = 24
    batch_size = 32

    # Create tf.data.Datasets
    train_dataset = create_dataset(train_data, window_size)
    train_dataset = train_dataset.shuffle(1000).batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = create_dataset(val_data, window_size)
    val_dataset = val_dataset.batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)

    # Reshape inputs to (batch, timesteps, features)
    train_dataset = train_dataset.map(
        lambda x, y: (tf.expand_dims(x, -1), y))
    val_dataset = val_dataset.map(
        lambda x, y: (tf.expand_dims(x, -1), y))

    # Build model
    print("Building model...")
    model = build_model(window_size)
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    # Train model
    print("Training model...")
    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on validation set
    print("Evaluating model...")
    val_loss, val_mae = model.evaluate(val_dataset)
    print("Validation MSE: {:.6f}".format(val_loss))
    print("Validation MAE: {:.6f}".format(val_mae))

    # Save model
    model.save('btc_forecast_model.keras')
    print("Model saved to btc_forecast_model.keras")

    return history, model


if __name__ == '__main__':
    forecast()
