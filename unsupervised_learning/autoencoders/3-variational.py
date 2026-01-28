#!/usr/bin/env python3
"""
Variational Autoencoder module
"""

import tensorflow.keras as keras
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Parameters:
    input_dims (int): dimensions of the input
    hidden_layers (list): number of nodes for each hidden encoder layer
    latent_dims (int): dimensions of the latent space

    Returns:
    encoder (keras.Model): encoder model (z, mean, log variance)
    decoder (keras.Model): decoder model
    auto (keras.Model): full variational autoencoder
    """

    # ---------- Encoder ----------
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(units=nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        mu, log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mu))
        return mu + keras.backend.exp(0.5 * log_var) * epsilon

    z = keras.layers.Lambda(sampling)([mu, log_var])

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[z, mu, log_var],
        name='encoder'
    )

    # ---------- Decoder ----------
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(units=nodes, activation='relu')(x)

    decoder_output = keras.layers.Dense(
        units=input_dims, activation='sigmoid')(x)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output,
        name='decoder'
    )

    # ---------- Autoencoder ----------
    auto_input = encoder_input
    z, mu, log_var = encoder(auto_input)
    reconstructed = decoder(z)

    auto = keras.Model(
        inputs=auto_input,
        outputs=reconstructed,
        name='variational_autoencoder'
    )

    # ----- VAE Loss -----
    reconstruction_loss = keras.losses.binary_crossentropy(
        auto_input, reconstructed
    )
    reconstruction_loss *= input_dims

    kl_loss = -0.5 * keras.backend.sum(
        1 + log_var - keras.backend.square(mu) -
        keras.backend.exp(log_var),
        axis=1
    )

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    auto.compile(optimizer='adam')

    return encoder, decoder, auto
