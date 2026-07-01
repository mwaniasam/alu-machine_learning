#!/usr/bin/env python3
"""
Performs the expectation maximization for a GMM
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    l_prev = 0
    b = False

    for i in range(iterations):
        g, lh = expectation(X, pi, m, S)
        if g is None or lh is None:
            return None, None, None, None, None

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(lh, 5)))

        if i > 0 and abs(lh - l_prev) <= tol:
            b = True
            break

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        l_prev = lh

    if not b:
        g, lh = expectation(X, pi, m, S)
        if g is None or lh is None:
            return None, None, None, None, None

        if verbose and iterations % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                iterations, round(lh, 5)))
        elif verbose:
            print("Log Likelihood after {} iterations: {}".format(
                iterations, round(lh, 5)))
    else:
        if verbose and i % 10 != 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(lh, 5)))

    return pi, m, S, g, lh
