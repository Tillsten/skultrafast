# -*- coding: utf-8 -*-
"""
Contains functions to find the time-zero and to interpolate the data.
"""

import numpy as np
import dv
import scipy.ndimage as nd
from statsmodels.api import RLM
import plot_funcs as pl

class est(object):
    pass

@dv.add_to_cls(est)
def use_gaussian(dat, sigma=1):
    """
    Use convolution with the derivate of an gaussian.
    """
    derivate = nd.gaussian_filter(dat, (sigma, 0), 1)
    return np.argmax(np.abs(derivate), 0)

@dv.add_to_cls(est)
def use_diff(dat, smooth=0):
    """
    Use numerical diff.
    """
    if smooth != 0:
        print smooth
        dat = nd.gaussian_filter(dat, smooth)
    derivate = np.diff(dat, 1, 0)
    return np.argmax(np.abs(derivate), 0)

@dv.add_to_cls(est)
def use_sv_filter(dat, window=7, polydeg=5):
    """
    Use savitzky-golay derivate.
    """
    out = np.zeros((dat.shape[1]))
    for i in range(dat.shape[1]):
        idx = np.argmax(dv.savitzky_golay(dat[:, i], window, polydeg, 1))
        out[i] = idx
    return out

@dv.add_to_cls(est)
def use_max(dat):
    """
    Uses the absolute maximum of the signal
    """
    return np.argmax(np.abs(dat), 0)
    
@dv.add_to_cls(est)
def use_first_abs(dat, val=5):
    """
    Returns the first index where dat>val.
    """
    idx = np.abs(dat) > val
    return np.argmax(idx, 0)

def robust_fit_tz(wl, tn, degree=3):
    """
    Apply a robust 3-degree fit to given tn-indexs.
    """    
    powers = np.arange(degree+1)
    X = wl[:,None] ** powers[None, :]
    r = RLM(tn, X).fit()
    return r.predict(), r.params[::-1]    


import ransac

def find_time_zero(d, value, method='abs', polydeg=3, *args):
    """
    Fits the dispersion of the timezero-point with a polynom of given
    degree. Uses the ransac algorithm to fit a approximation won by given
    method. The default method is to find the first point where the absolute 
    value is above a given limit.
    """
    
    class PoloynomialLeastSquaredModel:
        """
        Model used for ransac, fits data to polynom of specified degree.
        """
    
        def __init__(self, deg):        
            self.degree=deg
        
        def fit(self, data):
            x=data[:,0]
            y=data[:,1]
            return np.polyfit(x, y, self.degree)
    
        def get_error(self, data, model):
            x=data[:,0]
            y=data[:,1]
            return (y-np.poly1d(model)(x))**2

    t=d.t
    w=d.wl
    dat=d.data

    
    if method == 'abs':
        tn=t[np.argmin((np.abs(dat) < value),0)]
    
    ransac_data= np.column_stack((w,tn))    
    ransac_model = PoloynomialLeastSquaredModel(polydeg)
    m = ransac.ransac(ransac_data, ransac_model, tn.size * 0.2,
                      500, 0.1, tn.size * 0.6)    
    return np.poly1d(m)(w), m
    
def interpol(tup, tn, shift=0., new_t=None):
    """
    Uses linear interpolation to shift each channcel by given tn.
    """    
    dat = tup.data
    t = tup.t
    if new_t is None:
        new_t = t

    #t_array = np.tile(t.reshape(t.size, 1), (1, dat.shape[1]))
    t_array = t[:, None] - tn[None, :]    
    t_array -= shift
    dat_new = np.zeros((new_t.size, dat.shape[1]))
    for i in range(dat.shape[1]):        
        dat_new[:,i] = np.interp(new_t, t_array[:,i], dat[:,i], left=0)
    return dv.tup(tup.wl, t, dat_new)
    

def get_tz_cor(tup, method=use_diff, deg=3, plot=False,**kwargs):
    """
    Fully automatic timezero correction.
    """
    idx = method(tup.data, **kwargs)
    raw_tn = tup.t[idx]
    fit, p = robust_fit_tz(tup.wl, raw_tn, deg)
    #dv.subtract_background(tup.data, tup.t, fit, 400)
    cor = interpol(tup, fit)
    if plot:
        pl._plot_zero_finding(tup, raw_tn, fit, cor)
    return cor, fit
    
    
    
