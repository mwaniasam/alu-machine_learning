#!/usr/bin/env python3
"""
Calculates a GMM from a dataset
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset
    """
    gmm_obj = sklearn.mixture.GaussianMixture(n_components=k)
    gmm_obj.fit(X)

    pi = gmm_obj.weights_
    m = gmm_obj.means_
    S = gmm_obj.covariances_
    clss = gmm_obj.predict(X)
    bic = gmm_obj.bic(X)

    return pi, m, S, clss, bic
