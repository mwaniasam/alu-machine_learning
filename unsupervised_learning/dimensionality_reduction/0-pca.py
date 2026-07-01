#!/usr/bin/env python3
"""
Performs PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset
    """
    U, S, Vh = np.linalg.svd(X)
    cum_var = np.cumsum(S) / np.sum(S)

    r = np.argwhere(cum_var >= var)[0, 0]

    W = Vh[:r + 1].T

    return W
