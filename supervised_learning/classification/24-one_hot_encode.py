#!/usr/bin/env python3
"""Module for one-hot encoding"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix

    Args:
        Y: numpy.ndarray with shape (m,) containing numeric class labels
        classes: maximum number of classes found in Y

    Returns:
        One-hot encoding of Y with shape (classes, m), or None on failure
    """
    try:
        # Get number of examples
        m = Y.shape[0]
        
        # Create zero matrix
        one_hot = np.zeros((classes, m))
        
        # Set appropriate positions to 1
        one_hot[Y, np.arange(m)] = 1
        
        return one_hot
    except Exception:
        return None
