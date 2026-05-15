#!/usr/bin/env python3
"""
Module containing the LSTMCell class
"""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
        class constructor
        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        # Forget gate
        self.Wf = np.random.normal(size=(h + i, h))
        self.bf = np.zeros((1, h))

        # Update (input) gate
        self.Wu = np.random.normal(size=(h + i, h))
        self.bu = np.zeros((1, h))

        # Intermediate cell state
        self.Wc = np.random.normal(size=(h + i, h))
        self.bc = np.zeros((1, h))

        # Output gate
        self.Wo = np.random.normal(size=(h + i, h))
        self.bo = np.zeros((1, h))

        # Output projection
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step
        x_t: shape (m, i) input data
        h_prev: shape (m, h) previous hidden state
        c_prev: shape (m, h) previous cell state
        Returns: h_next, c_next, y
        """
        # Concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # 1. Forget Gate (sigmoid)
        f_t = 1 / (1 + np.exp(-(np.matmul(concat, self.Wf) + self.bf)))

        # 2. Update Gate (sigmoid)
        u_t = 1 / (1 + np.exp(-(np.matmul(concat, self.Wu) + self.bu)))

        # 3. Intermediate cell state (tanh)
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        # 4. Next Cell State
        c_next = f_t * c_prev + u_t * c_tilde

        # 5. Output Gate (sigmoid)
        o_t = 1 / (1 + np.exp(-(np.matmul(concat, self.Wo) + self.bo)))

        # 6. Next Hidden State
        h_next = o_t * np.tanh(c_next)

        # 7. Output y (softmax)
        z_y = np.matmul(h_next, self.Wy) + self.by
        exp_y = np.exp(z_y)
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, c_next, y
