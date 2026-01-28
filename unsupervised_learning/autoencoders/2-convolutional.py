#!/usr/bin/env python3
"""
Convolutional Autoencoder module
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder

    Parameters:
    input_dims (tuple): dimensions of the input
    filters (list): number of filters for each encoder conv layer
    latent_dims (tuple): dimensions of the latent space

    Returns:
    encoder (keras.Model): encoder model
    decoder (keras.Model): decoder model
    auto (keras.Model): full autoencoder model
    """

    # ---------- Encoder ----------
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(x)

    encoder = keras.Model(
        inputs=encoder_input, outputs=x, name='encoder')

    # ---------- Decoder ----------
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    rev_filters = filters[::-1]

    # All but last two conv layers
    for f in rev_filters[:-2]:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Second-to-last conv (valid padding, no upsampling)
    x = keras.layers.Conv2D(
        filters=rev_filters[-2],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(x)

    # Last conv (sigmoid, no upsampling)
    decoder_output = keras.layers.Conv2D(
        filters=input_dims[2],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

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
