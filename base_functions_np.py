# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:38:47 2013

@author: Tillsten
"""
import numpy as np


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
    ws = w
    k = 1 / (tau[..., None, None])
    t = (tt + tz).T[None, ...]
    y = np.exp(k * (ws * ws * k / (4.0) - t))
    y *= 0.5 * erfc(-t / ws + ws * k / (2.0))
    return y.T

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
    y:  ndarray (len(t), 4)
        Array containing a gaussian and it the scaled derivatives,
        each in its own column.
    """
    w = w / sq2
    tt = t + tz
    y = np.where(tt/w < 4, exp(-0.5 * (tt / w) ** 2) / (w * sqrt(2 * 3.14159265)), 0)
    y = np.tile(y[..., None], (1, 1, 4))
    y[..., 1] *= (-tt / w ** 2)
    y[..., 2] *= (tt ** 2 / w ** 4 - 1 / w ** 2)
    y[..., 3] *= (-tt ** 3 / w ** 6 + 3 * tt / w ** 4)
    y /= np.max(np.abs(y), 0)
    return y
