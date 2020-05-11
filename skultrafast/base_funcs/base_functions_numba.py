# -*- coding: utf-8 -*-
"""
Numba implementation of the base matrix building functions.
"""
import numpy as np
from numba import  vectorize, njit, jit, prange
import math
#from lmmv
sq2 = math.sqrt(2)


@jit
def _coh_gaussian(ta, w, tz):
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
    y:  ndarray (shape(t), 3)
        Array containing a gaussian and it the scaled derivatives,
        each in its own column.
    """

    w = w / 1.4142135623730951
    n, m = ta.shape
    y = np.zeros((n, m, 3))

    if tz != 0:
        ta = ta - tz

    _coh_loop(y, ta, w, n, m)
    #y_n = y / np.max(np.abs(y), 0)
    return y


exp_half = np.exp(0.5)


@njit(parallel=True)
def _coh_loop(y, ta, w, n, m):
    for i in prange(n):
        for j in prange(m):
            tt = ta[i, j]
            if tt / w < 3.:
                y[i, j, 0] = np.exp(
                    -0.5 * (tt / w) *
                    (tt / w))  # / (w * np.sqrt(2 * 3.14159265))
                y[i, j, 1] = y[i, j, 0] * (-tt * exp_half / w)
                y[i, j, 2] = y[i, j, 0] * (tt * tt / w / w - 1)
                #y[i, j, 2] = y[i, j, 0] * (-tt ** 3 / w ** 6 + 3 * tt / w ** 4)


@njit(parallel=True, fastmath=True)
def _fold_exp_and_coh(t_arr, w, tz, tau_arr):
    a = _fold_exp(t_arr, w, tz, tau_arr)
    b = _coh_gaussian(t_arr, w, tz)
    return a, b


@njit
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
    bot = 1 + a1 * x + a2 * x * x + a3 * x * x * x + a4 * x * x * x * x
    ret = 1. / (bot * bot * bot * bot)

    if smaller:
        ret = -ret + 2.

    return ret


@njit(fastmath=True, parallel=True)
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
    if t < -5. * w:
        return 0.
    elif t < 5. * w:
        #print -t/w + w*k/2., w, k, t
        return np.exp(
            k * (w * w * k / 4.0 - t)) * 0.5 * fast_erfc(-t / w + w * k / 2.)
    elif t > 5. * w:
        return np.exp(k * (w * w * k / (4.0) - t))


@njit
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
    y: ndarray(N, M, K)
       Folded exponentials for given taus.
    """
    n, m = t_arr.shape
    if w != 0:
        l = tau_arr.size
        out = np.empty((l, m, n))
        _fold_exp_loop(out, tau_arr, t_arr, tz, w, l, m, n)
        return out.T
    else:
        k = -1 / tau_arr
        out = np.exp((t_arr.reshape(n, m, 1) - tz) * k.reshape(1, 1, -1))
        return out


@njit(fastmath=True)
def _fold_exp_loop(out, tau_arr, t_arr, tz, w, l, m, n):
    for tau_idx in range(l):
        k = 1 / tau_arr[tau_idx]
        for j in range(m):
            for i in range(n):
                t = t_arr[i, j] - tz
                if t < -5. * w:
                    ret = 0
                elif t < 5. * w:
                    ret = np.exp(k *
                                 (w * w * k / 4.0 -
                                  t)) * 0.5 * fast_erfc(-t / w + w * k / 2.)
                elif t > 5. * w:
                    ret = np.exp(k * (w * w * k / (4.0) - t))
                out[tau_idx, j, i] = ret


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
    if not tz == 0:
        t_arr -= tz
    return np.exp(-rates * t_arr.T[None, ...]).T
