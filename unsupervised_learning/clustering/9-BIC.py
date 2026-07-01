#!/usr/bin/env python3
"""
Finds the best number of clusters for a GMM using
the Bayesian Information Criterion
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using
    the Bayesian Information Criterion
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0 or X.shape[0] < kmax:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    best_k = None
    best_result = None
    best_bic = float('inf')
    l_list = []
    b_list = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, lh = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None or m is None or S is None or g is None or lh is None:
            return None, None, None, None

        p = (k * d) + (k * d * (d + 1) / 2) + k - 1
        bic = p * np.log(n) - 2 * lh

        l_list.append(lh)
        b_list.append(bic)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    l_arr = np.array(l_list)
    b_arr = np.array(b_list)

    return best_k, best_result, l_arr, b_arr
