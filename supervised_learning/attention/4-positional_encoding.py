#!/usr/bin/env python3
"""
Module to calculate positional encoding for a Transformer model.
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding matrix.

    Parameters:
    - max_seq_len: integer, maximum sequence length
    - dm: integer, model depth (dimensionality of embedding space)

    Returns:
    - a numpy.ndarray of shape (max_seq_len, dm) containing the
      positional encoding vectors
    """
    # Initialize the encoding matrix with zeros
    PE = np.zeros((max_seq_len, dm))

    # Generate an array of positions (shape: max_seq_len, 1)
    position = np.arange(max_seq_len)[:, np.newaxis]

    # Generate the denominator step values for even indices 2i
    # shape: (dm / 2,)
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))

    # Apply sine to even indices (0, 2, 4, ...) in the matrix
    PE[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices (1, 3, 5, ...) in the matrix
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
