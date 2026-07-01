#!/usr/bin/env python3
"""
Performs agglomerative clustering on a dataset
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    plt.figure()
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()

    return clss
