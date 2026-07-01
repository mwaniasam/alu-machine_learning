#!/usr/bin/env python3
"""
Calculates the maximization step in the EM algorithm for a GMM
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    if not np.isclose(np.sum(g, axis=0), np.ones(n)).all():
        return None, None, None

    pi = np.sum(g, axis=1) / n
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        m[i] = np.sum(g[i, :, np.newaxis] * X, axis=0) / np.sum(g[i])
        diff = X - m[i]
        S[i] = np.matmul(g[i] * diff.T, diff) / np.sum(g[i])

    return pi, m, S
