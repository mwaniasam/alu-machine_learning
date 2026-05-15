#!/usr/bin/env python3
"""
Module containing the deep_rnn function
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN
    rnn_cells: list of RNNCell instances of length l
    X: data of shape (t, m, i)
    h_0: initial hidden state of shape (l, m, h)
    Returns: H, Y
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]

    # Initialize H with shape (t + 1, l, m, h)
    # H[0] will store the initial states for all layers
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        # The input for the first layer is the data X
        current_input = X[step]

        for layer in range(l):
            # Forward through the specific cell for this layer
            # Each layer uses its own previous hidden state H[step, layer]
            h_next, y = rnn_cells[layer].forward(H[step, layer], current_input)

            # Store the hidden state for the next time step
            H[step + 1, layer] = h_next

            # The input for the NEXT layer is the hidden state of this layer
            current_input = h_next

        # The output Y is the output of the LAST layer
        Y[step] = y

    return H, Y
