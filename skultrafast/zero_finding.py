# -*- coding: utf-8 -*-
"""
Contains functions to find the time-zero and to interpolate the data.
"""

import numpy as np
import skultrafast.dv as dv
import scipy.ndimage as nd
from statsmodels.api import RLM, robust

from matplotlib.pyplot import plot
#from skultrafast.fitter import _coh_gaussian
from scipy.linalg import lstsq
#from scipy.optimize import leastsq

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
def use_max(dat, use_abs=True):
    """
    Uses the absolute maximum of the signal
    """
    if use_abs:
        dat = np.abs(dat)
    return np.argmax(dat, 0)

@dv.add_to_cls(est)
def use_first_abs(dat, val=5):
    """
    Returns the first index where abs(dat)>val.
    """
    idx = np.abs(dat) > val
    return np.argmax(idx, 0)

import scipy.optimize as opt

@dv.add_to_cls(est)
def use_fit(dat, t, tau=[ 5, 20000], w0=0.08, tn=None, n=-1):
    """
    Fits each transient with only w and x0 free.
    """
    out = np.zeros(dat.shape[1])
    w_out = np.zeros(dat.shape[1])
    t = t[:n]
    o = tn[0]
    w = w0
    for i in range(dat.shape[1]):
        y = dat[:n, i]
        f = lambda p: _fit_func(t, y, -p[0], p[1], tau)
        f_sum = lambda p: (f(p)**2).sum()
        
        try:
            if not np.isnan(o) and False:
                k = o + np.diff(tn)[i]

            else:
                k = tn[i]
                w = w0
            
            #o, w = leastsq(f, list([k, w0]))[0][:2]
            # = opt.minimize(f_sum, [k,w], method='BFGS')
            #x = cma.fmin(f_sum, [o, w0], 0.03, bounds=[(0,0.04),(5, 0.2)], restarts=1, verb_log=0)
            x = opt.brute(f_sum, (range((tn-0.1),(tn+0.1),0.01),
                                  np.range(0.04,0.13,0.01)))
            o, w =x[0]
            if abs(o-tn[i]) > 0.04:
                plot(t, f([o, w]) + y)
                plot(t, y, 'o')
        except NameError:
            o = w =  np.NaN

        out[i] = o
        w_out[i] = w
    return out, w_out


def _fit_func(t, y, x0, w, tau):
    """
    Fit
    """
    base = np.column_stack((_fold_exp(t, w, x0, np.array(tau)).T,#))
                            _coh_gaussian(t, w, x0)))
    base = np.nan_to_num(base)
    c = lstsq(base, y[:, None])
    y_fit = np.dot( base, c[0])
    return (y_fit[:,0] - y)



def robust_fit_tz(wl, tn, degree=3, t=1.345):
    """
    Apply a robust 3-degree fit to given tn-indexs.
    """
    powers = np.arange(degree+1)
    X = wl[:,None] ** powers[None, :]
    norm = robust.norms.HuberT(t=t)
    r = RLM(tn, X, M=norm).fit()
    return r.predict(), r.params[::-1]



from skultrafast import ransac

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
    no_nan = ~np.any(np.isnan(tup.data), 0)
    fit, p = robust_fit_tz(tup.wl[no_nan], raw_tn[no_nan], deg)
    #dv.subtract_background(tup.data, tup.t, fit, 400)
    fit = np.polyval(p, tup.wl)
    cor = interpol(tup, fit)
    if plot:
        from . import plot_funcs as pl
        pl._plot_zero_finding(tup, raw_tn, fit, cor)
    return cor, fit

if __name__ == '__main__':
    a = np.load('SD5039.npz')
    w, t, d = a['arr_0']
    d -= d[:10,:].mean(0)
    d = d[:,20:]
    w = w[20:]
    t = t
    nt, tn = get_tz_cor(dv.tup(w, t, d), use_max)
    n = 5
    figure(0)
    k, j = use_fit(d[:, ::n], t, tn=tn[::n], n=100)
    figure(1)
    pcolormesh(w, t, d)
    plot(w[::n], k[::])
    plot(w[::n], tn[::n])
    ylim(2, 5)
    xlim(w.min(), w.max())


