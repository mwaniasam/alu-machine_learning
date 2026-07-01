#!/usr/bin/env python3
"""
Calculates the total intra-cluster variance for a data set
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    try:
        distances_sq = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)
        min_distances_sq = np.min(distances_sq, axis=1)
        var = np.sum(min_distances_sq)
        return var
    except Exception:
        return None
