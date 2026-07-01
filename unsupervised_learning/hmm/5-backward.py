#!/usr/bin/env python3
"""
The Backward Algorithm
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape

    if Transition.shape != (N, N) or Initial.shape != (N, 1):
        return None, None

    B = np.zeros((N, T))
    B[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        term = Emission[:, Observation[t + 1]] * B[:, t + 1]
        B[:, t] = np.matmul(Transition, term)

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B
