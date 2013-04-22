# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:03:26 2013

@author: Tillsten
"""
import numpy as np


from numba import autojit, vectorize, f8, jit
from math import exp, erfc, sqrt
sq2 = sqrt(2)

@autojit
def _coh_gaussian(t, w, tz):
    """
    Models coherent artifacts proportional to a gaussian and it's first three derivatives.

    Parameters
    ----------
    t:  ndarray
        2d - Array containing the time-coordinates
    w:  float
        The assumed width/sq2
    tz: float
        The assumed time zero.

    Returns
    -------
    y:  ndarray (shape(t), 4)
        Array containing a gaussian and it the scaled derivatives,
        each in its own column.
    """

    w = w / 1.4142135623730951
    ta = t + tz
    n, m = ta.shape
    y = np.zeros(( n, m, 4))
    for i in range(n):
        for j in range(m):
            tt = ta[i, j]
            if tt/w < 2.5:
                y[i, j, 0] = np.exp(-0.5 * (tt / w)* (tt / w)) / (w * np.sqrt(2 * 3.14159265))
                y[i, j, 1] = y[i, j, 0] * (-tt / w / w)
                y[i, j, 2] = y[i, j, 0] * (tt * tt / w / w  / w / w - 1 / w /w)
                y[i, j, 3] = y[i, j, 0] * (-tt ** 3 / w ** 6 + 3 * tt / w ** 4)

    y /= np.max(np.abs(y), 0)
    return y


@jit('f8(f8)', nopython=True)
def fast_erfc(x):
    """
    Calculates the erfc near zero faster than
    the libary function, but has a bigger error, which
    is not a problem for us.

    Parameters
    ----------
    x: float
        The array

    Returns
    -------
    ret: float
        The erfc of x.
    """
    a1 = 0.278393
    a2 = 0.230389
    a3 = 0.000972
    a4 = 0.078108
    smaller = x < 0
    if smaller:
        x = x * -1.
    bot = 1 + a1*x + a2*x*x +a3*x*x*x + a4*x*x*x*x
    ret = 1./(bot*bot*bot*bot)

    if smaller:
        return -ret + 2.
    else:
        return  ret


@jit('f8(f8, f8, f8, f8)', nopython=True)
def folded_fit_func(t, tz, w, k):
    """
    Returns the value of a folded exponentials.
    Employs some domain checking for making the calculation.

    Parameters
    ----------
    t: float
        The time.
    tz: float
        Timezero.
    w:
        Width of the gaussian system response.
    k:
        rate of the decay.
    """
    t = t - tz
    if t < -2.5 * w:
        return 0.
    elif t < 2.5 * w:
        #print -t/w + w*k/2., w, k, t
        return exp(k * (w*w*k/4.0 - t)) * 0.5 * fast_erfc(-t/w + w*k/2.)
    elif t < 5./k:
        return exp(k* (w*w*k/ (4.0) - t))
    else:
        return 0.

@jit(f8[:, :, :], [f8[:, :], f8, f8, f8[:]])
def _fold_exp(t_arr, w, tz, tau_arr):
    """
    Returns the values of the folded exponentials for given parameters.

    Parameters
    ----------
    t_arr:  ndarray(N, M)
        Array containing the time-coordinates
    w:  float
        The assumed width/sq2
    tz: float
        The assumed time zero.
    tau_arr: ndarray(K)
        The M-decay rates.

    Returns
    -------
    y: ndarray
       Folded exponentials for given taus.
    """
    n, m = t_arr.shape
    l = tau_arr.size
    out = np.empty((l, m, n))
    for tau_idx in range(l):
        k = 1 / tau_arr[tau_idx]
        for j in range(m):
            for i in range(n):
                out[tau_idx, j, i] = folded_fit_func(t_arr[i, j], -tz, w, k)
    return out.T


#jit(f8[:, :, :], [f8[:, :], f8, f8, f8[:]])
def _exp(t_arr, w, tz, tau_arr):
    """
    Returns the values of exponentials for given parameters.

    Parameters
    ----------
    t_arr:  ndarray(N, M)
        Array containing the time-coordinates
    w:  float
        The assumed width/sq2, not used.
    tz: float
        The assumed time zero.
    tau_arr: ndarray(K)
        The M-decay rates.

    Returns
    -------
    y: ndarray(N,M, K)
       Exponentials for given taus and t_array.
    """
    rates = 1 / tau_arr[:, None, None]
    if not tz==0:
        t_arr -= tz
    return np.exp(-rates * t_arr.T[None, ...]).T


def calc_gaussian_fold(y_arr, sigma, slice_to_fold, slice_to_calc):
    """
    Folds the data with an gaussian.

    Parameters
    ----------
    y_arr: ndarray(N,M)
        array of the data to be folded

#














    w: float
        the width of the gaussian.

    slice_to_fold: (int, int)
        slice where the folding is inserted

    slice_to_calculate: (int, int)
        slice where the folding is calculated, should
        be larger than the fold_slice to offset border effects

    Returns
    -------
    folded_y: ndarray(N, M)
        The folded array.

    """


    import scipy.ndimage as nd
    to_fold = y_arr[slice_to_calc[0]:slice_to_calc[1], :]
    folded = nd.gaussian_filter1d(to_fold, sigma, axis=0, mode='constant')
    idx_low = slice_to_fold[0] - slice_to_calc[0]
    idx_high = slice_to_calc[0] - slice_to_fold[0]
    y_arr[slice_to_fold[0]:slice_to_fold[1], :] = folded[idx_low:idx_high, :]
    return y_arr

