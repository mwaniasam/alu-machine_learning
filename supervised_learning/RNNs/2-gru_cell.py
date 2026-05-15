#!/usr/bin/env python3
"""
Module containing the GRUCell class
"""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit (GRU) cell"""

    def __init__(self, i, h, o):
        """
        class constructor
        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        # Update gate weights and biases
        self.Wz = np.random.normal(size=(h + i, h))
        self.bz = np.zeros((1, h))

        # Reset gate weights and biases
        self.Wr = np.random.normal(size=(h + i, h))
        self.br = np.zeros((1, h))

        # Intermediate (candidate) hidden state weights and biases
        self.Wh = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))

        # Output weights and biases
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        x_t: numpy.ndarray of shape (m, i) with input data
        h_prev: numpy.ndarray of shape (m, h) with previous hidden state
        Returns: h_next, y
        """
        # Concatenate previous hidden state and input for gates
        concatenated = np.concatenate((h_prev, x_t), axis=1)

        # 1. Update Gate (z_t) uses sigmoid
        z_t = 1 / (1 + np.exp(-(np.matmul(concatenated, self.Wz) + self.bz)))

        # 2. Reset Gate (r_t) uses sigmoid
        r_t = 1 / (1 + np.exp(-(np.matmul(concatenated, self.Wr) + self.br)))

        # 3. Intermediate Hidden State (h_tilde)
        # It uses the reset gate to "forget" parts of h_prev
        reset_h_prev = r_t * h_prev
        concat_reset = np.concatenate((reset_h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_reset, self.Wh) + self.bh)

        # 4. Next Hidden State (h_next)
        # Linear interpolation between h_prev and h_tilde using z_t
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # 5. Output (y) uses softmax
        z_y = np.matmul(h_next, self.Wy) + self.by
        exp_y = np.exp(z_y)
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y
