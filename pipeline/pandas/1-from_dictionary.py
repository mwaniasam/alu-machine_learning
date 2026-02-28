#!/usr/bin/env python3
"""Module for creating a pd.DataFrame from a np.ndarray"""
import pandas as pd


def from_numpy(array):
    """Creates a pd.DataFrame from a np.ndarray with alphabetical column labels"""
    cols = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=cols)