# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as st
import scipy.signal as sig
from collections import namedtuple
from scipy.constants import c, physical_constants


tup = namedtuple('tup','wl t data')

def add_to_cls(cls):
    def function_enum(fn):
        setattr(cls, fn.__name__, staticmethod(fn))
        return fn
    return function_enum

def add_tup(str_lst):
    def function_enum(fn):
        str_lst[0].append(fn.__name__)
        str_lst[1].append(fn)
        return fn
    return function_enum

def fs2cm(t):
    return 1/(t * 3e-5)

def cm2fs(cm):
    return  1/(cm * 3e-5)

def nm2cm(nm):
    return 1e7/nm

def cm2nm(cm):
    return 1e7/cm

def cm2eV(cm):
    eV_m = physical_constants['electron volt-inverse meter relationship'][0]
    eV_cm = eV_m/100
    return cm/eV_cm

def eV2cm(eV):
    eV_m = physical_constants['electron volt-inverse meter relationship'][0]
    eV_cm = eV_m/100
    return eV*eV_cm

def cm2THz(cm):
    return c/(cm*100)

def trimmed_mean(arr, axis=-1, ratio=2., use_sem=True):
    arr = np.sort(arr, axis=axis)
    std = np.std(arr, axis, keepdims=1)
    std = np.std(st.trimboth(arr, 0.1, axis), keepdims=1)
    mean = np.mean(st.trimboth(arr, 0.1, axis), keepdims=1)
    #std = np.std(st.trimboth(arr, 0.1, axis), keepdims=1)
    #mean = np.mean(st.trimboth(arr, 0.1, axis), keepdims=1)
    idx = np.abs(arr - mean) > ratio * std
    n = np.sqrt(np.sum(~idx, axis))
    if not use_sem:
        n = 1
    arr[idx] = np.nan

    mean = np.nanmean(arr, axis)
    std = np.nanstd(arr, axis, ddof=1)/n
    return mean, std


from scipy.interpolate import UnivariateSpline

def smooth_spline(x, y, s):
    s = UnivariateSpline(x, y, s=s)
    return s(x)

def svd_filter(d, n=6):
    u, s, v = np.linalg.svd(d, full_matrices=0)
    s[n:] = 0
    f = np.dot(u, np.diag(s).dot(v))
    return f

def apply_spline(t, d, s=None):
    out = np.zeros_like(d)
    for i in range(d.shape[1]):
        out[:, i] =smooth_spline(t, d[:, i], s)
    return out

def normalize(x):
    return x/abs(x).max()

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
    binned_std = np.empty_like(binned)
    binned_wl = np.empty(n)
    for i in range(n):
        data = dat[:,idx[i]:idx[i+1]]
        weights = 1/std[:,idx[i]:idx[i+1]]**2
        binned[:,i] = np.average(data, 1, weights)
        binned_std[:, i] = np.average(std[:,idx[i]:idx[i+1]], 1, weights)
        binned_wl[i] = np.mean(wl[idx[i]:idx[i+1]])
    return binned, binned_wl, binned_std


def binner(n, wl, dat, func=np.mean):
    """
    Given wavelengths and data it bins the data into n-wavelenths.
    Returns bdata and bwl

    """
    i = np.argsort(wl)
    wl = wl[i]
    dat = dat[:, i]
    idx=np.searchsorted(wl,np.linspace(wl.min(),wl.max(),n+1))
    binned=np.empty((dat.shape[0], n))
    binned_wl=np.empty(n)
    for i in range(n):
        binned[:,i]=func(dat[:,idx[i]:idx[i+1]],1)
        binned_wl[i]=np.mean(wl[idx[i]:idx[i+1]])
    return binned, binned_wl

def fi(w, x):
    """
    Given a value, it finds the index of the nearest value in the array.

    Parameters
    ----------
    w : np.ndarray
        Array where to look.
    x : float or list of floats
        Value or values to look for.

    Returns
    -------
    int or list of ints
        Indicies of the nearest values.

    """
    try:
        len(x)
    except TypeError:
        x = [x]
    ret =  [np.argmin(np.abs(w-i)) for i in x]
    if len(ret)==1:
        return ret[0]
    else:
        return ret

def subtract_background(dat, t, tn, offset=0.3):
    out = np.zeros_like(dat)
    for i in range(dat.shape[1]):
        mask = (t-tn[i]) < -offset
        corr = dat[mask, i].mean()
        out[:, i] = dat[:, i] -  corr
    return out

def polydetrend(x, t=None, deg=3):
    if t is None:
        t = np.arange(x.shape[0])
    p = np.polyfit(t, x, deg)
    yf = np.poly1d(p)(t)
    return x - yf

def exp_detrend(y, t, start_taus=[1], use_constant=True):
    m, yf = exp_fit(t, y, start_taus, use_constant=use_constant, verbose=0)
    return y - yf

def arr_polydetrend(x, t=None, deg=3):
    out = np.zeros_like(x)
    for i in range(x.shape[1]):
        out[:, i] = polydetrend(x[:, i], t, deg)
    return out

from scipy.stats import trim_mean
def meaner(dat, t, llim, ulim, proportiontocut=0.0):
    return trim_mean(dat[fi(t, llim):fi(t, ulim)],  axis=0, proportiontocut=proportiontocut)

def legend_format(l):
    return [str(i/1000.)+ ' ps' for i in l]


def apply_sg(y, window_size, order, deriv=0):
    out = np.zeros_like(y)
    coeffs = sig.savgol_coeffs(window_size, order, deriv=0, use='dot')
    for i in range(y.shape[1]):
        out[:, i] = coeffs.dot(y[:, i])
    return out

import scipy.ndimage as nd
def apply_sg_scan(y, window_size, order, deriv=0):
    out = np.zeros_like(y)
    c = sig.savgol_coeffs(window_size, order, deriv=0)
#    for s in range(y.shape[-1]):
#        for i in range(y.shape[1]):
#            print c.shape
    out = nd.convolve1d(y, c, 0, mode='nearest')
     #out, s] = c.dot(y[:, i, 1, s])
    return out

def calc_error(args):
    """
    Calculates the error from a leastsq fit infodict.
    """
    p, cov, info, mesg, success = args
    chisq = sum(info["fvec"] * info["fvec"])
    dof = len(info["fvec"]) - len(p)
    sigma = np.array([np.sqrt(cov[i, i]) * np.sqrt(chisq / dof) for i in range(len(p))])
    return p, sigma

def min_pulse_length(width_in_cm, shape='gauss'):
    width_hz = width_in_cm * 3e10
    if shape == 'gauss':
        return (0.44 / width_hz) / 1e-15


def wavelength2rgb(w):
    """
    Converts a wavelength to a RGB color.
    """
    if w >= 380 and w < 440:
        R = -(w - 440.) / (440. - 350.)
        G = 0.0
        B = 1.0
    elif w >= 440 and w < 490:
        R = 0.0
        G = (w - 440.) / (490. - 440.)
        B = 1.0
    elif w >= 490 and w < 510:
        R = 0.0
        G = 1.0
        B = -(w - 510.) / (510. - 490.)
    elif w >= 510 and w < 580:
        R = (w - 510.) / (580. - 510.)
        G = 1.0
        B = 0.0
    elif w >= 580 and w < 645:
        R = 1.0
        G = -(w - 645.) / (645. - 580.)
        B = 0.0
    elif w >= 645 and w <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    if (w>700.):
        s=.3+.7* (780.-w)/(780.-700.)
    elif (w<420.):
        s=.3+.7*(w-380.)/(420.-380.)
    else:
        s=1.
    R,G,B=(np.array((R,G,B))*s)**0.8
    return R,G,B

def equal_color(plots1,plots2):
    if len(plots1)!=len(plots2):
        raise ValueError
    for (plot1,plot2) in zip(plots1,plots2):
        plot2.set_color(plot1.get_color())


def find_linear_part(t):
    """
    Finds the first value of an 1d-array where the difference betweeen
    consecutively value varys.
    """
    d=np.diff(t)
    return np.argmin(np.abs(d-d[0])<0.00001)

def rebin(a, new_shape):
    """
    Resizes a 2d array by averaging or repeating elements,
    new dimensions must be integral factors of original dimensions

    Parameters
    ----------
    a : array_like
    Input array.
    new_shape : tuple of int
    Shape of the output array

    Returns
    -------
    rebinned_array : ndarray
    If the new shape is smaller of the input array, the data are averaged,
    if the new shape is bigger array elements are repeated

    See Also
    --------
    resize : Return a new array with the specified shape.

    Examples
    --------
    >>> a = np.array([[0, 1], [2, 3]])
    >>> b = rebin(a, (4, 6)) #upsize
    >>> b
    array([[0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [2, 2, 2, 3, 3, 3],
    [2, 2, 2, 3, 3, 3]])

    >>> c = rebin(b, (2, 3)) #downsize
    >>> c
    array([[ 0. , 0.5, 1. ],
    [ 2. , 2.5, 3. ]])

    """
    M, N = a.shape
    m, n = new_shape
    if m<M:
        return a.reshape((m,M/m,n,N/n)).mean(3).mean(1)
    else:
        return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)

from scipy.sparse.linalg import svds
def efa(dat, n, reverse=False):
    """
    Doing evolving factor analyis.
    """
    if reverse:
        data=dat[::-1, :]
    else:
        data=dat

    out=np.zeros((data.shape[0], n))
    for i in range(6, data.shape[0]):
        sv = svds(data[:i, :], min(i,n))[1]
        out[i, :] = sv
    return out

def moving_efa(dat, n, ncols, method='svd'):
    out=np.zeros((dat.shape[0], n))
    p = PCA()
    for i in range(0, dat.shape[0]-n):
        if method=='svd':
            sv = np.linalg.svd(dat[i:i+ncols,:])[1][:n]
        elif method=='pca':
            p = p.fit(dat[i:i+ncols,:])
            sv = p.explained_variance_ratio_[:n]
        out[i,:] = sv
    return out

from scipy.optimize import nnls

def pfid_tau_to_w(tau):
    """
    Given the rise time of the pertuabed
    free induction decay, calculate the
    corresponding spectral width in cm-^1.
    """
    return 1/(np.pi*3e7*tau*1e-9)

def als(dat, n=5):
    u, s, v = np.linalg.svd(dat)
    u0=u[:n]
    v0=v.T[:n]
    u0 = np.random.random(u0.shape)+0.1
    v0 = np.random.random(v0.shape)+0.1
    res = np.linalg.lstsq(u0.T,dat)[1].sum()
    res = 10000.
    for i in range(15000):
        if i % 2 == 0:
            v0 = do_nnls(u0.T, dat)
            v0 /= np.max(v0, 1)[:, None]
            res_n = ((u0.T.dot(v0) - dat)**2).sum()
        else:

            u0, res_n, t , t = np.linalg.lstsq(v0.T, dat.T)
            ##r.fit(dat.T, v0.T)
            #u0 = r.coef_[:]
            res_n = ((u0.T.dot(v0) - dat)**2).sum()

            if abs(res-res_n) < 0.001:
                break
            else:
                print(i, res_n)
                res = res_n
    return u0.T, v0.T

def spec_int(tup, r, is_wavelength=True):
    wl, t, d = tup.wl, tup.t, tup.data
    if is_wavelength:
        wl = 1e7/wl
        wl1, wl2 = 1e7 / r[0], 1e7 / r[1]
    else:
        wl1, wl2 = r
    ix = np.argsort(wl)
    wl = wl[ix]
    d = d[:, ix]

    idx1, idx2 = sorted([fi(wl, wl1), fi(wl, wl2)])
    dat = np.trapz(d[:, idx1:idx2], wl[idx1:idx2]) / np.ptp(wl[idx1:idx2])
    return dat

#import mls
def do_nnls(A,b):
    n = b.shape[1]
    out = np.zeros((A.shape[1], n))
    for i in range(n):
        #mls.bounded_lsq(A.T, b[:,i], np.zeros((A.shape[1],1)), np.ones((A.shape[1],1))).shape
        out[:,i] =  nnls(A, b[:,i])[0]
    return out

import lmfit
def exp_fit(x, y, start_taus = [1], use_constant=True, amp_max=None, amp_min=None, weights=None, amp_penalty=0,
            verbose=True, start_amps=None):
    num_exp = len(start_taus)
    para = lmfit.Parameters()
    if use_constant:
        para.add('const', y[-1] )

    for i in range(num_exp):
        para.add('tau' + str(i), start_taus[i], min=0)
        y_c = y - y[-1]
        if start_amps is None:
            a = y_c[fi(x, start_taus[i])]
        else: 
            a = start_amps[i]
        para.add('amp' + str(i), a)
        if amp_max is not None:
            para['amp' + str(i)].max = amp_max
        if amp_min is not None:
            para['amp' + str(i)].min = amp_min

    def fit(p):
        y_fit = np.zeros_like(y)
        if use_constant:
            y_fit += p['const'].value

        for i in range(num_exp):
            amp = p['amp'+str(i)].value
            tau = p['tau'+str(i)].value

            y_fit += amp * np.exp(-x/tau) 

        return y_fit

    def res(p):
        
        if weights is None:
            pen = 0
            for i in range(num_exp):
                pen += p['amp'+str(i)].value**2
            return np.hstack(((y - fit(p)), amp_penalty*pen))
        else:
            return (y - fit(p)) / weights

    mini = lmfit.minimize(res, para)
    if verbose:
        lmfit.report_fit(mini)
    y_fit = fit(mini.params)
    return mini, y_fit

def calc_ratios(fitter, tmin=0.35, tmax=200):
    from skultrafast import zero_finding
    tup = zero_finding.interpol(fitter, fitter.tn)
    w, t, d = tup
    i = fi(t, tmin)
    i_max = fi(t, tmax)
    t = t[i:i_max]
    d = d[i:i_max, :]

    pos = np.where(d > 0, d, 0)
    neg = np.where(d < 0, d, 0)
    pos = np.trapz(pos, w)
    neg = np.trapz(neg, w)
    return t, pos, neg, pos/neg, d.sum(1)


def make_fi(data_to_search):
    return lambda x: fi(data_to_search, x)
