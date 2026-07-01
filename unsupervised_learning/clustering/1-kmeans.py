#!/usr/bin/env python3
"""
Performs K-means on a dataset
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(min_vals, max_vals, size=(k, X.shape[1]))

    for i in range(iterations):
        C_copy = np.copy(C)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        for j in range(k):
            if len(X[clss == j]) == 0:
                C[j] = np.random.uniform(min_vals, max_vals,
                                         size=(1, X.shape[1]))
            else:
                C[j] = np.mean(X[clss == j], axis=0)

        if np.array_equal(C_copy, C):
            break

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)
    return C, clss
