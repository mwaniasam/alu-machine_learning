#!/usr/bin/env python3
"""
Performs PCA on a dataset
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(ndim, int) or ndim <= 0 or ndim > X.shape[1]:
        return None

    X_m = X - np.mean(X, axis=0)
    U, S, Vh = np.linalg.svd(X_m)
    W = Vh[:ndim].T
    T = np.matmul(X_m, W)

    return T
