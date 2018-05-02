# -*- coding: utf-8 -*-
"""
This module contains various filters and binning methods. All take
a dv tup and return a tup.
"""

from . import dv
import numpy as np
import scipy.ndimage as nd
import scipy.signal as sig

def svd_filter(tup, n=6):
    """
    Only use the first n-components.

    Parameters
    ----------
    tup:
        data object.
    n:
        number of svd components used.
    """
    wl, t, d = tup.wl, tup.t, tup.data
    u, s, v = np.linalg.svd(d, full_matrices=0)
    s[n:] = 0
    f = np.dot(u, np.diag(s).dot(v))
    return dv.tup(wl, t, f)

def wiener(tup, size=(3,3), noise=None):
    wl, t, d = tup.wl, tup.t, tup.data
    f = sig.wiener(d, size, noise=noise)
    return dv.tup(wl, t, f)

def uniform_filter(tup, sigma=(2, 2)):
    """
    Apply an uniform filter to data.
    """
    wl, t, d = tup.wl, tup.t, tup.data
    f = nd.uniform_filter(d, size=sigma, mode="nearest")
    return dv.tup(wl, t, f)


def gaussian_filter(tup, sigma=(2, 2)):
    """
    Apply an uniform filter to data.
    """
    wl, t, d = tup.wl, tup.t, tup.data
    f = nd.gaussian_filter(d, sigma=sigma, mode="nearest")
    return dv.tup(wl, t, f)

def sg_filter(tup, window_length=11, polyorder=2, deriv=0, axis=0):
    """
    Apply a Savitzky-Golay filter to a tup.

    Parameter
    ---------
    tup:
        Data
    window_length:
        Winodw length of the filter
    polyorder:
        The order of local polynomial. Must be smaller than window_length.
    deriv:
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0.

    """
    wl, t, d = tup.wl, tup.t, tup.data
    f = sig.savgol_filter(d, window_length, polyorder, axis=axis,
                          mode='nearest')
    return dv.tup(wl, t, f)

def bin_channels(tup, n=200, method=np.mean):
    """
    Bin the data onto n-channels.
    """

    def binner(n, wl, dat):
        """
        Given wavelengths and data it bins the data into n-wavelenths.
        Returns bdata and bwl

        """
        i = np.argsort(wl)
        wl = wl[i]
        dat = dat[:, i]
        idx = np.searchsorted(wl,np.linspace(wl.min(),wl.max(),n+1))
        binned = np.empty((dat.shape[0], n))
        binned_wl = np.empty(n)
        for i in range(n):
            binned[:,i] = method(dat[:,idx[i]:idx[i+1]],1)
            binned_wl[i] = np.mean(wl[idx[i]:idx[i+1]])
        return binned, binned_wl


    wl, t, d = tup.wl, tup.t, tup.data
    binned_d, binned_wl = binner(n, wl, d)
    return dv.tup(binned_wl, t, binned_d)

def weighted_binner(n, wl, dat, std):
    """
    Given wavelengths and data it bins the data into n-wavelenths.
    Returns bdata and bwl

    """
    i = np.argsort(wl)
    wl = wl[i]
    dat = dat[:, i]
    idx = np.searchsorted(wl,np.linspace(wl.min(),wl.max(),n+1))
    binned = np.empty((dat.shape[0], n))
    binned_wl = np.empty(n)
    for i in range(n):
        data = dat[:,idx[i]:idx[i+1]]
        weights = 1/std[:,idx[i]:idx[i+1]]
        binned[:,i] = np.average(data, 1, weights)
        binned_wl[i] = np.mean(wl[idx[i]:idx[i+1]])
    return binned, binned_wl

def cut_tup(tup, from_t=-1e6, to_t=1e6, from_wl=-1e6, to_wl=1e6):
    wl, t, d = tup.wl.copy(), tup.t.copy(), tup.data.copy()
    t0, t1 = dv.fi(t, from_t), dv.fi(t, to_t)
    w0, w1 = dv.fi(wl, from_wl), dv.fi(wl, to_wl)
    w0, w1 = min(w0, w1), max(w0, w1)    
    return dv.tup(wl[w0:w1], t[t0:t1], d[t0:t1, w0:w1])


def norm_tup(tup, min_div=3):
    wl, t, d = tup.wl, tup.t, tup.data
    np.max(d)
