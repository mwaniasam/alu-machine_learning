#!/usr/bin/env python3
"""
The Forward Algorithm
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model
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

    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        F[:, t] = np.matmul(F[:, t - 1], Transition) * Emission[:, Observation[t]]

    P = np.sum(F[:, T - 1])
    return P, F
