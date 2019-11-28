"""Module with various utility functions. Was called dv in older Versions."""
import numpy as np
from scipy.special import erf


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
    for _ in range(max_iter):
        median = np.ma.median(data, axis, keepdims=1)
        std = np.ma.std(data, axis, keepdims=1)
        upper, lower = median + sigma * std, median - sigma * std
        data = np.ma.masked_greater(data, upper, copy=False)
        data = np.ma.masked_less(data, lower, copy=False)
        n = data.mask.sum()
        if n == num_masked:
            break
        else:
            num_masked = n
    return data


def gauss_step(x, amp: float, center: float, sigma: float):
    """Returns the stepfunction (erf-style) for given arguments.

    Parameters
    ----------
    x : array
        Independent variable
    amp : float
        Amplitude of the step
    center : float
        Position of the step
    sigma : float
        Width of the step

    Returns
    -------
    array
        The step functions
    """
    return amp * 0.5 * (1 + erf((x - center) / sigma / np.sqrt(2)))
