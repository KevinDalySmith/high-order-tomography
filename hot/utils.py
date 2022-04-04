
import numpy as np
import itertools
from itertools import combinations


def bootstrap(X, n_groups, sample_size=None, seed=None):
    """
    Bootstrap a sample.

    Parameters
    ----------
    X : np.ndarray
        (N, n) array with N observations of n variates.
    n_groups : int
        Number of groups to generate by re-sampling.
    sample_size : int, optional, default=None
        Number of observations in each group.
        If None, resamples are the same size as the original sample.
    seed : int, optional, default=None
        Random seed.

    Returns
    -------
    Y : np.ndarray
        (sample_size, n, n_groups) array of bootstrapped data.
    """
    if sample_size is None:
        sample_size = X.shape[0]
    _, n = X.shape
    Y = np.ndarray([sample_size, n, n_groups])
    np.random.seed(seed)
    for i in range(n_groups):
        sample_idx = np.random.choice(sample_size, size=sample_size, replace=True)
        Y[..., i] = X[sample_idx, :]
    return Y


def power_set(S):
    """
    Iterator over the power set of S (excluding the empty set).
    """
    return itertools.chain.from_iterable(combinations(S, r) for r in range(1, len(S) + 1))
