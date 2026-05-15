#!/usr/bin/env python3
"""
Module containing the BidirectionalCell class
"""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """
        class constructor
        i: dimensionality of the data
        h: dimensionality of the hidden states
        o: dimensionality of the outputs
        """
        # Hidden states weights/biases for forward direction
        self.Whf = np.random.normal(size=(h + i, h))
        self.bhf = np.zeros((1, h))

        # Hidden states weights/biases for backward direction
        self.Whb = np.random.normal(size=(h + i, h))
        self.bhb = np.zeros((1, h))

        # Output weights and biases
        # Input is 2 * h due to concatenation of forward/backward states
        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Calculates the hidden state in the forward direction"""
        concatenated = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concatenated, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """Calculates the hidden state in the backward direction"""
        concatenated = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(concatenated, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """
        Calculates all outputs for the RNN
        H: numpy.ndarray of shape (t, m, 2 * h) containing concatenated
           hidden states from both directions
        Returns: Y, the outputs of shape (t, m, o)
        """
        t, m, double_h = H.shape
        o = self.Wy.shape[1]

        # Initialize output array
        Y = np.zeros((t, m, o))

        for step in range(t):
            # linear transformation: z = H * Wy + by
            z = np.matmul(H[step], self.Wy) + self.by

            # Softmax activation
            exp_z = np.exp(z)
            Y[step] = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return Y
