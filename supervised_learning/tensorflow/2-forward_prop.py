#!/usr/bin/env python3
"""Creates the forward propagation graph for a neural network"""

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation for the neural network

    Arguments:
        x (tf.Tensor): input placeholder
        layer_sizes (list): list of number of nodes per layer
        activations (list): list of activation functions per layer

    Returns:
        tf.Tensor: output prediction of the network
    """
    output = x

    for n, activation in zip(layer_sizes, activations):
        output = create_layer(output, n, activation)

    return output
