#!/usr/bin/env python3
"""
Module containing the RNNCell class
"""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """
        class constructor
        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        # Wh and bh are for the concatenated hidden state and input data
        # The concatenated input has shape (m, h + i)
        # Therefore, Wh must be (h + i, h) to produce h_next (m, h)
        self.Wh = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))

        # Wy and by are for the output
        # Wy must be (h, o) to produce y (m, o)
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        x_t: numpy.ndarray of shape (m, i) with input data
        h_prev: numpy.ndarray of shape (m, h) with previous hidden state
        Returns: h_next, y
        """
        # Concatenate h_prev and x_t along the columns (axis 1)
        # Resulting shape: (m, h + i)
        concatenated = np.concatenate((h_prev, x_t), axis=1)

        # Calculate next hidden state: h_next = tanh(concat * Wh + bh)
        h_next = np.tanh(np.matmul(concatenated, self.Wh) + self.bh)

        # Calculate output: y = softmax(h_next * Wy + by)
        z = np.matmul(h_next, self.Wy) + self.by
        
        # Softmax implementation
        exp_z = np.exp(z)
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, y

