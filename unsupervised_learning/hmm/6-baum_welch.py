#!/usr/bin/env python3
"""
The Baum-Welch Algorithm
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model
    """
    if not isinstance(Observations, np.ndarray) or \
            len(Observations.shape) != 1:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    T = Observations.shape[0]
    M, N_out = Emission.shape

    if Transition.shape != (M, M) or Initial.shape != (M, 1):
        return None, None

    for _ in range(iterations):
        # Forward
        F = np.zeros((M, T))
        F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
        for t in range(1, T):
            F[:, t] = np.matmul(F[:, t - 1],
                                Transition) * Emission[:, Observations[t]]

        # Backward
        B = np.zeros((M, T))
        B[:, T - 1] = 1
        for t in range(T - 2, -1, -1):
            B[:, t] = np.matmul(Transition,
                                Emission[:, Observations[t + 1]] * B[:, t + 1])

        # Xi and Gamma
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            term = np.matmul(Transition,
                                 Emission[:, Observations[t + 1]] * B[:, t + 1])
            denominator = np.sum(F[:, t] * term)
            if denominator == 0:
                return None, None
            for i in range(M):
                numerator = F[i, t] * Transition[i, :] * \
                    Emission[:, Observations[t + 1]] * B[:, t + 1]
                xi[i, :, t] = numerator / denominator

        gamma = np.zeros((M, T))
        for t in range(T - 1):
            gamma[:, t] = np.sum(xi[:, :, t], axis=1)

        gamma[:, T - 1] = F[:, T - 1] * B[:, T - 1]
        gamma_sum = np.sum(gamma[:, T - 1])
        if gamma_sum == 0:
            return None, None
        gamma[:, T - 1] /= gamma_sum

        # Update Parameters
        Transition_new = np.sum(xi, axis=2) / np.sum(gamma[:, :-1],
                                                     axis=1).reshape((M, 1))

        Emission_new = np.zeros((M, N_out))
        for k in range(N_out):
            mask = (Observations == k)
            Emission_new[:, k] = (np.sum(gamma[:, mask], axis=1) /
                                  np.sum(gamma, axis=1))

        if np.allclose(Transition, Transition_new) and \
                np.allclose(Emission, Emission_new):
            return Transition_new, Emission_new

        Transition = Transition_new
        Emission = Emission_new

    return Transition, Emission
