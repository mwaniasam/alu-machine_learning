#!/usr/bin/env python3
"""
Module containing the rnn function
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN
    rnn_cell: instance of RNNCell to be used for forward propagation
    X: data to be used, numpy.ndarray of shape (t, m, i)
    h_0: initial hidden state, numpy.ndarray of shape (m, h)
    Returns: H, Y
    """
    t, m, i = X.shape
    m, h = h_0.shape
    o = rnn_cell.by.shape[1]

    # Initialize H and Y with zeros
    # H includes h_0, so it has t + 1 time steps
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Set the first hidden state as h_0
    H[0] = h_0

    # Loop through each time step
    for step in range(t):
        # rnn_cell.forward returns (h_next, y)
        # Use the previous hidden state from H
        h_next, y = rnn_cell.forward(H[step], X[step])

        # Store results
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
