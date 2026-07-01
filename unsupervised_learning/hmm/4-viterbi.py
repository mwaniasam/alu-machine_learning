#!/usr/bin/env python3
"""
The Viterbi Algorithm
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states
    for a hidden markov model
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

    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)

    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    B[:, 0] = 0

    for t in range(1, T):
        for j in range(N):
            trans_probs = V[:, t - 1] * Transition[:, j]
            V[j, t] = np.max(trans_probs) * Emission[j, Observation[t]]
            B[j, t] = np.argmax(trans_probs)

    P = np.max(V[:, T - 1])
    best_last = np.argmax(V[:, T - 1])

    path = [int(best_last)]
    for t in range(T - 1, 0, -1):
        best_last = B[best_last, t]
        path.append(int(best_last))

    path.reverse()
    return path, P
