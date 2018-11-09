"""Module with various utility functions. Was called dv in older Versions."""
import numpy as np


def sigma_clip(data, sigma=3, max_iter=5, axis=-1):
    """Masks outliers by iteratively removing points outside given
    standard deviations.

    Parameters
    ----------
    data : np.ndarray
        The data array.
    sigma : float
        All data outside +/- sigma*std will be masked in each iteration.
    max_iter : int
        How many iterations are done. If a new iteration does not mask new
        values, the function will break the loop.

    Returns
    -------
    np.ma.MaskedArray
        Array with outliers being masked.
    """
    data = np.ma.masked_invalid(data)
    num_masked = 0
    for i in range(max_iter):
        median = np.ma.median(data, axis, keepdims=1)
        std = np.ma.std(data, axis, keepdims=1)
        upper, lower = median + sigma*std, median - sigma*std
        data = np.ma.masked_greater(data, upper, copy=False)
        data = np.ma.masked_less(data, lower, copy=False)
        n = data.mask.sum()
        if n == num_masked:
            break
        else:
            num_masked = n
    return data
