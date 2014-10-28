# -*- coding: utf-8 -*-
"""
This module contains various filters and binning methods. All take
a dv tup and return a tup.
"""

import dv
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
    wl, t, d = tup
    u, s, v = np.linalg.svd(d, full_matrices=0)
    s[n:] = 0
    f = np.dot(u, np.diag(s).dot(v))
    return dv.tup(wl, t, f)
    
def uniform_filter(tup, sigma=(2, 2)):
    """
    Apply an uniform filter to data.
    """
    wl, t, d = tup
    f = nd.uniform_filter(d, mode="nearest")
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
    wl, t, d = tup
    f = sig.savgol_filter(d, window_length, polyorder, axis=axis, 
                          mode='nearest')
    return dv.tup(wl, t, f)
    
def bin_channels(tup, n=200):
    """
    Bin the data onto n-channels.
    """
    wl, t, d = tup
    binned_d, binned_wl = dv.binner(n, wl, d)
    return dv.tup(binned_wl, t, binned_d)


