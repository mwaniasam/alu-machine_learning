#!/usr/bin/env python3
"""
Calculates the probability density function of a Gaussian distribution
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    d = m.shape[0]

    det = np.linalg.det(S)
    inv = np.linalg.inv(S)

    norm_const = 1.0 / (np.sqrt(((2 * np.pi) ** d) * det))

    diff = X - m

    exponent = -0.5 * np.sum(np.matmul(diff, inv) * diff, axis=1)

    pdf_values = norm_const * np.exp(exponent)

    pdf_values = np.maximum(pdf_values, 1e-300)

    return pdf_values
