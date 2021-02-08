"""Module with various utility functions. Was called dv in older Versions."""
import numpy as np
from scipy.special import erf
from scipy.stats import median_absolute_deviation
from .unit_conversions import cm2THz

import functools
import wrapt


def weighted_binning(x, arr, bins, weights=None):
    """
    Bins a 1D array to given bins using weights.
    """
    weights_total, _ = np.histogram(x, bins, weights=weights)
    if weights is None:
        weights = 1
    binned_total, _ = np.histogram(x, bins, weights=arr*weights)
    return binned_total / weights_total


def simulate_binning(wrapped=None, *, fac=5):
    """
    Simulates
    """
    if wrapped is None:
        return functools.partial(simulate_binning,
                fac=fac)

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        wl = kwargs['wl']
        n = fac * wl.size
        dx = abs(wl[1] - wl[0])
        mids = (wl[:1] + wl[1:]) / 2.
        upsampled = np.linspace(wl.min()-dx, wl.max()+dx, n)
        idx = np.digitize(upsampled, mids)
        kwargs['wl'] = upsampled
        counts = np.bincount(idx)
        result = wrapped(*args, **kwargs)
        return np.bincount(idx, result)/counts

    return wrapper(wrapped)


def sigma_clip(data, sigma=3, max_iter=5, axis=-1, use_mad=False):
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
        if use_mad:
            std = median_abs_deviation(data, axis=1)
        else:
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
    om_21 = cm2THz(np.asarray(om_21)) * 2 * np.pi

    T, om, om_10 = np.meshgrid(T, om, om_10, indexing='ij', copy=False)
    om_21 = np.broadcast_to(om_21, om_10.shape)
    T_2 = np.broadcast_to(T_2, om_10.shape)
    dom = om - om_10
    dom2 = om - om_21

    num = (1/T_2) * np.cos(dom * T) - dom2 * np.sin(dom * T)
    return np.exp(-T / T_2) * num / (dom2**2 + (1 / T_2**2))


def pfid(T, om, om_10, fac, om_21, T_2):
    om = cm2THz(om) * 2 * np.pi
    om_10 = cm2THz(np.asarray(om_10)) * 2 * np.pi
    om_21 = cm2THz(np.asarray(om_21)) * 2 * np.pi

    T, om, om_10 = np.meshgrid(T, om, om_10, indexing='ij', copy=False)
    om_21 = np.broadcast_to(om_21, om_10.shape)
    T_2 = np.broadcast_to(T_2, om_10.shape)
    fac = np.broadcast_to(fac, om_10.shape)
    dom = om - om_10
    dom2 = om - om_21

    # tmp to make it faster
    dec = np.exp(-T / T_2)
    cos = np.cos(dom * T)
    sin = np.sin(dom * T)


    num = (1/T_2) * cos - dom * sin
    r4 = dec * num / (dom**2 + (1 / T_2**2))    
    
    num = (1/T_2) * cos - dom2 * sin
    r6 = dec * num / (dom2**2 + (1 / T_2**2))
    return r4 + fac*r6

def linreg_std_errors(A, y):
    """
    Calculates the solution and error terms in a linear regression.

    Parameters
    ----------
    A : ndarray
        Basis matrix
    y : ndarray
        Data
    Returns
    -------
    (ndarray, ndarray, ndarray,)
        Tuple of three arrays: standard error, variance matrix, r2
    """
    x = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ x[0] 
    resi = y - fit
    r2 = 1 - x[1] / (y.shape[0] * y.var(0))
    vcv = A.T @ A
    epsvar = np.var(resi, axis=0, ddof=2)
    bvar = np.linalg.inv(vcv) * epsvar[:, None, None]
    bstd = np.zeros_like(x[0])
    for i in range(bstd.shape[1]):
        bstd[:, i] = np.sqrt(np.diag(bvar[i]))
    return bstd, bvar, r2
