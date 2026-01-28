#!/usr/bin/env python3
"""
Sparse Autoencoder module
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder

    Parameters:
    input_dims (int): dimensions of the input
    hidden_layers (list): number of nodes in each hidden encoder layer
    latent_dims (int): dimensions of the latent space
    lambtha (float): L1 regularization parameter for sparsity

    Returns:
    encoder (keras.Model): encoder model
    decoder (keras.Model): decoder model
    auto (keras.Model): sparse autoencoder model
    """

    # ---------- Encoder ----------
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(units=nodes, activation='relu')(x)

    latent = keras.layers.Dense(
        units=latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)

    encoder = keras.Model(
        inputs=encoder_input, outputs=latent, name='encoder')

    # ---------- Decoder ----------
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(units=nodes, activation='relu')(x)

    decoder_output = keras.layers.Dense(
        units=input_dims, activation='sigmoid')(x)

    decoder = keras.Model(
        inputs=decoder_input, outputs=decoder_output, name='decoder')

    # ---------- Autoencoder ----------
    auto_input = encoder_input
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto = keras.Model(
        inputs=auto_input, outputs=decoded, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
