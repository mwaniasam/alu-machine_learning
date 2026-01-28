#!/usr/bin/env python3
"""Creates a neural network layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a dense layer for a neural network

    Arguments:
        prev (tf.Tensor): tensor output from previous layer
        n (int): number of nodes in the layer
        activation (function): activation function for the layer

    Returns:
        tf.Tensor: output of the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer"
    )

    return layer(prev)
