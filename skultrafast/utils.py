"""Module with various utility functions. Was called dv in older Versions."""
import numpy as np
from scipy.special import erf
from .unit_conversions import cm2THz


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
        upper, lower = median + sigma*std, median - sigma*std
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
    return amp * 0.5 * (1 + erf((x-center) / sigma / np.sqrt(2)))


def pfid_r4(T, om, om_10, T_2):
    """
    Calculates the PFID contribution for pure bleaching.

    See the PFID tutorial for a longer explanation. The function does
    broadcasting, hence it is possible to calculate the PFID contributions of
    serveral bands at once. For that, om_10 and T_2 must have the same shape.

    Parameters
    ----------
    T : 1D-ndarry
        Delays between pump and probe. The formula assume a postive 
        delays.
    om : 1D-ndarray
        Array of frequencies given in wavenumbers (cm-1).
    om_10 : 1D-ndarray or float
        Frequencies of the ground-state absorbtions
    T_2 : 1D_ndarray or float
        Decoherence time of the bands.

    Returns
    -------
    ndarry        
    """
    om = cm2THz(om) * 2 * np.pi
    om_10 = cm2THz(np.asarray(om_10)) * 2 * np.pi

    T, om, om_10 = np.meshgrid(T, om, om_10, indexing='ij', copy=False)
    T_2 = np.broadcast_to(T_2, om_10.shape)
    dom = om - om_10

    num = (1/T_2) * np.cos(dom * T) - dom * np.sin(dom * T)
    return np.exp(-T / T_2) * num / (dom**2 + (1 / T_2**2))


def pfid_r6(T, om, om_10, om_21, T_2):
    """
    Calculates the PFID contribution for the shifted frequecy.

    See the PFID tutorial for a longer explanation. The function does
    broadcasting, hence it is possible to calculate the PFID contributions of
    serveral bands at once. For that, om_10, om_21 and T_2 must have the same
    shape.

    Parameters
    ----------
    T : 1D-ndarry
        Delays between pump and probe. The formula assume a postive 
        delays.
    om : 1D-ndarray
        Array of frequencies given in wavenumbers (cm-1).
    om_10 : 1D-ndarray or float
        Frequencies of the ground-state absorbtions
    om_21 : 1D-ndarray or float
        Frequencies of the shifted frequency
    T_2 : 1D_ndarray or float
        Decoherence time of the bands.

    Returns
    -------
    ndarry        
    """

    om = cm2THz(om) * 2 * np.pi
    om_10 = cm2THz(np.asarray(om_10)) * 2 * np.pi
    om_21 = cm2THz(np.asarray(om_10)) * 2 * np.pi

    T, om, om_10 = np.meshgrid(T, om, om_10, indexing='ij', copy=False)
    om_21 = np.broadcast_to(om_21, om_10.shape)
    T_2 = np.broadcast_to(T_2, om_10.shape)
    dom = om - om_10
    dom2 = om - om_21

    num = (1/T_2) * np.cos(dom * T) - dom2 * np.sin(dom * T)
    return np.exp(-T / T_2) * num / (dom2**2 + (1 / T_2**2))
