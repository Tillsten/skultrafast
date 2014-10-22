# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import splrep, splev
from collections import namedtuple

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

def trimmed_mean(arr, axis=-1, ratio=3.):
    std = np.std(arr, axis, keepdims=1)
    mean = np.mean(arr, axis, keepdims=1)  
    idx = np.abs(arr - mean) > 3. * std
    arr[idx] = np.nan
    return np.nansum(arr, axis=axis) / np.sum(np.isfinite(arr), axis=axis)


def dichro_to_angle(d):
    return np.arccos(np.sqrt((2*d-1)/(d+2)))/np.pi*180
def angle_to_dichro(x):
    return (1+2*np.cos(x)**2)/(2-np.cos(x)**2)

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
    
def binner(n, wl, dat):
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
        binned[:,i]=np.mean(dat[:,idx[i]:idx[i+1]],1)
        binned_wl[i]=np.mean(wl[idx[i]:idx[i+1]])
    return binned, binned_wl

def fi(w,x):
    """needs global w as array. gives idnex to nearest value"""
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
    t = t or np.arange(x.shape[0])
    p = np.polyfit(t, x, deg)
    yf = np.poly1d(p)(t)
    return x - yf
    
def arr_polydetrend(x, t=None, deg=3):
    out = np.zeros_like(x)
    for i in range(x.shape[1]):
        out[:, i] = polydetrend(x[:, i], t, deg)
    return out


def meaner(dat, t, llim, ulim):
    return np.mean(dat[fi(t, llim):fi(t, ulim)], 0)
    
def legend_format(l):
    return [str(i/1000.)+ ' ps' for i in l]

def savitzky_golay(y, window_size, order, deriv=0):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t,files=glob.glob(name+'*dat?.npy') np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannerypsd1D = radialProfile.azimuthalAverage(psd2D)
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m, y, mode='valid')

def apply_sg(y, window_size, order, deriv=0):
    out = np.zeros_like(y)
    for i in range(y.shape[1]):
        out[:, i] = savitzky_golay(y[:, i], window_size, order, deriv)
    return out
    
def apply_sg_scan(y, window_size, order, deriv=0):
    out = np.zeros_like(y)
    for s in range(y.shape[-1]):
        for i in range(y.shape[1]):
            out[:, i, s] = savitzky_golay(y[:, i, s], window_size, order, deriv)
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

#
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
        print sv
        out[i, :]=sv
    return out

def moving_efa(dat, n, ncols, method='svd'):
    out=np.zeros((dat.shape[0], n))
    #p=PCA()
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
    
print pfid_tau_to_w(1)
    
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
                print i, res_n
                res = res_n
    return u0.T, v0.T


#import mls
def do_nnls(A,b):
    n = b.shape[1]
    out = np.zeros((A.shape[1], n))
    for i in xrange(n):
        #mls.bounded_lsq(A.T, b[:,i], np.zeros((A.shape[1],1)), np.ones((A.shape[1],1))).shape
        out[:,i] =  nnls(A, b[:,i])[0]
    return out

import lmfit
def exp_fit(x, y, start_taus = [1], use_constant=True, amp_max=None):
    print y.dtype
    num_exp = len(start_taus)
    para = lmfit.Parameters()
    if use_constant:
        para.add('const', y[-1] )
    
    for i in range(num_exp):
        para.add('tau' + str(i), start_taus[i])
        y_c = y - y[-1]
        a = y_c[fi(x, start_taus[i])]        
        para.add('amp' + str(i), a)
        if amp_max is not None:
            para['amp' + str(i)].max = amp_max
            para['amp' + str(i)].min = -amp_max
        
    def fit(p):
        y_fit = np.zeros_like(y) 
        if use_constant:
            y_fit += p['const'].value
        
        for i in range(num_exp):
            amp = p['amp'+str(i)].value
            tau = p['tau'+str(i)].value
            
            y_fit += amp * np.exp(-x/tau)
        
        return y_fit
    fit(para)

    def res(p):
        return y - fit(p)        


    mini = lmfit.minimize(res, para)
    lmfit.report_errors(para)
    y_fit = fit(para)
    return mini, y_fit 

def calc_ratios(fitter, tmin=0.35, tmax=200):
    from skultrafast import zero_finding
    tup = zero_finding.interpol(fitter, fitter.tn)
    w, t, d = tup
    i = fi(t, tmin)
    i_max = fi(t, tmax)
    t = t[i:i_max]
    d = d[i:i_max, :]
    pos = np.where(d > 0, d, 0).sum(1)
    neg = np.where(d < 0, d, 0).sum(1)
    return t, pos, neg, pos/neg, d.sum(1)
        
        

#if __name__=='__main__':
#    import numpy as np
#    ss = apply_spline(t, d[..., 0], s=9)        
#    plot(ss[:, 18])
#    t, p, n, pn, total = calc_ratios(g)
#    m,yf = exp_fit(t, pn, [1, 11])
#    plot(t, pn)
#    plot(t, yf)
    #figure(1)
    #clf()
    #imshow(a-u.dot(v.T).T)
    #o = efa(a, 10)
    #o2 = efa(a,10,True)
    #plot(t,log(o),'k')
    #plot(t,log(o2[::-1,:]),'r')
    #u, s, v = np.linalg.svd(a[:,:].T)
    #o = moving_efa(a[:,:].T, 5, 8, 'pca')
    #lo = log(o)#/log(o[:,0:1])
    ##hlines(log(s[:8])/log(s[0]), wl.min(), wl.max())
    #plot(wl, lo)
    #show()