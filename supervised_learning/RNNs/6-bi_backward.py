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
        # (Concatenation of forward and backward states is 2 * h)
        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction
        for one time step
        """
        concatenated = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concatenated, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction
        for one time step
        x_t: shape (m, i) contains the data input for the cell
        h_next: shape (m, h) contains the next hidden state
        Returns: h_prev, the previous hidden state
        """
        # Concatenate the 'next' hidden state and the current input
        # Results in shape (m, h + i)
        concatenated = np.concatenate((h_next, x_t), axis=1)

        # Calculate the previous backward hidden state using tanh activation
        # h_prev = tanh(concatenated * Whb + bhb)
        h_prev = np.tanh(np.matmul(concatenated, self.Whb) + self.bhb)

        return h_prev
