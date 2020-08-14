"""
skstab - Clustering metrics

@author Florent Forest, Alex Mourer
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _contingency_matrix(y_true, y_pred):
    w = np.zeros((y_true.max() + 1, y_pred.max() + 1), dtype=np.int64)
    for c, k in zip(y_true, y_pred):
        w[c, k] += 1  # w[c, k] = number of c-labeled samples in cluster k
    return w


def clustering_accuracy(y_true, y_pred):
    """Unsupervised clustering accuracy.

    Can only be used if the number of target classes in y_true is equal to the number of clusters in y_pred.

    Parameters
    ----------
    y_true : array, shape = [n]
        true labels.
    y_pred : array, shape = [n]
        predicted cluster ids.

    Returns
    -------
    accuracy : float in [0,1] (higher is better)
        accuracy score.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    w = _contingency_matrix(y_true, y_pred).T
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def minimum_matching_distance(y_true, y_pred):
    """Minimum matching distance (MMD).

    Can only be used if the number of target classes in y_true is equal to the number of clusters in y_pred.

    Parameters
    ----------
    y_true : array, shape = [n]
        true labels.
    y_pred : array, shape = [n]
        predicted cluster ids.

    Returns
    -------
    mmd : float in [0,1]
        minimum matching distance.
    """
    return 1.0 - clustering_accuracy(y_true, y_pred)
