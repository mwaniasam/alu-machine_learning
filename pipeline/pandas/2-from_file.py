#!/usr/bin/env python3
"""Module for loading data from a file as a pd.DataFrame"""
import pandas as pd


def from_file(filename, delimiter):
    """Loads data from a file as a pd.DataFrame"""
    return pd.read_csv(filename, sep=delimiter)