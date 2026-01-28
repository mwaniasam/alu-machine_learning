#!/usr/bin/env python3
"""
Variational Autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): dimensionality of the input
        hidden_layers (list): number of nodes for each encoder hidden layer
        latent_dims (int): dimensionality of the latent space

    Returns:
        encoder (keras.Model): encoder model
        decoder (keras.Model): decoder model
        auto (keras.Model): full autoencoder model
    """

    # ---------- Encoder ----------
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        mu, log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mu))
        return mu + keras.backend.exp(0.5 * log_var) * epsilon

    z = keras.layers.Lambda(sampling)([mu, log_var])

    encoder = keras.Model(
        inputs=inputs,
        outputs=[z, mu, log_var],
        name='encoder'
    )

    # ---------- Decoder ----------
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(
        input_dims, activation='sigmoid')(x)

    decoder = keras.Model(
        inputs=latent_inputs,
        outputs=outputs,
        name='decoder'
    )

    # ---------- Autoencoder ----------
    z, mu, log_var = encoder(inputs)
    reconstructed = decoder(z)

    auto = keras.Model(
        inputs=inputs,
        outputs=reconstructed,
        name='variational_autoencoder'
    )

    # ---------- Loss ----------
    reconstruction_loss = keras.losses.binary_crossentropy(
        inputs, reconstructed
    )
    reconstruction_loss *= input_dims

    kl_loss = -0.5 * keras.backend.sum(
        1 + log_var
        - keras.backend.square(mu)
        - keras.backend.exp(log_var),
        axis=1
    )

    auto.add_loss(
        keras.backend.mean(reconstruction_loss + kl_loss)
    )

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
