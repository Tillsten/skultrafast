# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:38:47 2013

@author: Tillsten
"""
import numpy as np
from scipy.special import erfc
import math
sq2 = math.sqrt(2)


def _fold_exp(tt, w, tz, tau):
    """
    Returns the values of the folded exponentials for given parameters.

    Parameters
    ----------
    tt:  ndarray(N)
        Array containing the time-coordinates
    w:  float
        The assumed width/sq2
    tz: float
        The assumed time zero.
    tau: ndarray(M)
        The M-decay rates.

    Returns
    -------
    y: ndarray
       Folded exponentials for given taus.

    """
    if w == 0:
        k = -1 / tau
        return np.exp((tt.reshape(tt.shape[0], tt.shape[1], 1), -tz) * k.reshape(1, 1, -1))
    ws = w
    k = 1 / (tau[..., None, None])
    t = (tt + tz).T[None, ...]
    y = np.exp(k * (ws * ws * k / (4.0) - t))
    y *= 0.5 * erfc(-t / ws + ws * k / (2.0))
    return y.T


def _fold_exp_and_coh(t_arr, w, tz, tau_arr):
    a = _fold_exp(t_arr, w, tz, tau_arr)
    b = _coh_gaussian(t_arr, w, tz)
    return a, b


exp_half = np.exp(0.5)


def _coh_gaussian(t, w, tz):
    """Models artifacts proportional to a gaussian and it's derivatives

    Parameters
    ----------
    t:  ndarray
        Array containing the time-coordinates
    w:  float
        The assumed width/sq2
    tz: float
        The assumed time zero.

    Returns
    -------
    y:  ndarray (len(t), 3)
        Array containing a gaussian and it the scaled derivatives,
        each in its own column.
    """
    w = w / sq2
    tt = t + tz
    idx = (tt / w < 3.)
    y = np.where(idx, np.exp(-0.5 * (tt/w) * (tt/w)), 0)
    y = np.tile(y[..., None], (1, 1, 3))
    tt = tt[idx]
    y[idx, ..., 1] *= (-tt * exp_half / w)
    y[idx, ..., 2] *= (tt*tt/w/w - 1)
    #y[idx,..., 2] *= (-tt ** 3 / w ** 6 + 3 * tt / w ** 4)
    return y
