#!/usr/bin/env python3
"""
Performs K-means on a dataset using sklearn
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset using sklearn
    """
    kmeans_obj = sklearn.cluster.KMeans(n_clusters=k)
    kmeans_obj.fit(X)

    C = kmeans_obj.cluster_centers_
    clss = kmeans_obj.labels_

    return C, clss
