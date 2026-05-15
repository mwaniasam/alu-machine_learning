#!/usr/bin/env python3
"""
Module containing the bi_rnn function
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN
    bi_cell: instance of BidirectionalCell
    X: data of shape (t, m, i)
    h_0: initial forward hidden state, shape (m, h)
    h_t: initial backward hidden state, shape (m, h)
    Returns: H, Y
    """
    t, m, i = X.shape
    m, h = h_0.shape

    # Initialize arrays for forward and backward hidden states
    h_forward = np.zeros((t, m, h))
    h_backward = np.zeros((t, m, h))

    # Calculate Forward hidden states
    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        h_forward[step] = h_prev

    # Calculate Backward hidden states
    # We iterate backwards through the time steps
    h_next = h_t
    for step in range(t - 1, -1, -1):
        h_next = bi_cell.backward(h_next, X[step])
        h_backward[step] = h_next

    # Concatenate forward and backward states along the last dimension
    # Shape of H will be (t, m, 2 * h)
    H = np.concatenate((h_forward, h_backward), axis=2)

    # Use the cell's output method to compute Y from concatenated H
    Y = bi_cell.output(H)

    return H, Y
