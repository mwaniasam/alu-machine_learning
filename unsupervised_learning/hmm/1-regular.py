#!/usr/bin/env python3
"""
Regular Chains
"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    n = P.shape[0]
    if not np.allclose(np.sum(P, axis=1), 1):
        return None

    P_k = np.copy(P)
    is_reg = False
    for _ in range(n ** 2):
        if np.all(P_k > 0):
            is_reg = True
            break
        P_k = np.matmul(P_k, P)

    if not is_reg:
        return None

    evals, evecs = np.linalg.eig(P.T)
    idx = np.where(np.isclose(evals, 1))[0]
    if len(idx) == 0:
        return None

    steady_state = evecs[:, idx[0]].real
    steady_state = steady_state / np.sum(steady_state)
    return steady_state.reshape(1, n)
