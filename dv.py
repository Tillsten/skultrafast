# -*- coding: utf-8 -*-
from numpy import zeros, loadtxt, ceil, interp, around, arange, floor,log10
import numpy as np
from scipy.interpolate import splrep, splev
from collections import namedtuple

tup = namedtuple('Result','wl t data')


def binner(n, wl, dat):
    """ Given wavelengths and data it bins the data into n-wavelenths.
    """
    idx=np.searchsorted(wl,np.linspace(wl.min(),wl.max(),n+1))
    binned=np.empty((dat.shape[0], n))
    binned_wl=np.empty(n)
    for i in range(n):
        binned[:,i]=np.mean(dat[:,idx[i]:idx[i+1]],1)
        binned_wl[i]=np.mean(wl[idx[i]:idx[i+1]])
    return binned, binned_wl

def find_w(w,x):
    """needs global w as array. gives idnex to nearest value"""
    idx = (abs(w - x)).argmin()
    return idx


def first_min(sig, low_lim):
    i = 4
    while (not sig[i - 1] > sig[i] < sig[i + 1]) or sig[i] > low_lim:
        i += 1

        if i > sig.shape[0]-2:
            break
    return i


def get_abs(s, lim):
    z = zeros(s.shape[1], dtype = 'int')
    for i in range(s.shape[1]):
        j = 0
        while j < s.shape[0] - 2 and (s[j, i]) < lim:
            j = j + 1
        if j != s.shape[0] - 1: z[i] = j
    return z

def get_zeros(s, low_lim = -0.5):
    z = zeros(s.shape[1], dtype = 'int')
    for i in range(s.shape[1]):
        z[i] = first_min(s[:, i], low_lim)
    return z

def subtract_background(dat, t, tn, offset=0.3):
    for i in range(dat.shape[1]):
        mask=(t-tn[i])<-offset
        corr=dat[mask,i].mean()
        dat[:,i]-=corr
    return dat

def poly_time(k, t, level = 0.25, up = 2000, low = -500,w=np.arange(400),deg=2):
    k = t[get_zeros(k, level)]
    m = np.ma.masked_array(k, np.logical_or(k<low, k>up))
    masked_range = np.ma.masked_array(w, m.mask)
    p = np.polyfit(masked_range.compressed(), m.compressed(), deg)
    return(np.poly1d(p)(np.arange(k.shape[0]))),p

def poly_time_abs(k, t, level = -1, up = 2000, low = -1000,w=arange(400), deg=2):
    k = t[np.argmin(abs(k)<level,0)]
    m = np.ma.masked_array(k, np.logical_or(k<low, k>up))
    masked_range = np.ma.masked_array(w, m.mask)
    p = np.polyfit(masked_range.compressed(), m.compressed(), deg)
    return(np.poly1d(p)(w)),p

def poly_time_abs_r(k, t,w, level = -1, up = 2000, low = -1000):
    k = t[get_abs(k, level)]
    m = np.ma.masked_array(k, np.logical_or(k<low, k>up))
    masked_range = np.ma.masked_array(w, m.mask)
    p = np.polyfit(masked_range.compressed(), m.compressed(), 3)    
    return(np.poly1d(p)(arange(k.shape[0])))

def interpol(dat,t,time_zero,shift,new_t=np.array([-500,-250,0,250,500,1000,1500,2000,500000])):
    t_array=np.tile(t.reshape(t.size,1),(1,dat.shape[1]))
    dat_new=zeros((new_t.size,dat.shape[1]))
    for i in range(dat.shape[1]):
        t_array[:,i]-=time_zero[i]-shift
        dat_new[:,i]=interp(new_t,t_array[:,i],dat[:,i], left=0)
    return dat_new

def interpol_sm(r,t,re,time_zero,shift,new_t=np.array([-500,-250,0,250,500,1000,1500,2000,500000]),s=None):
    t_array=np.tile(t.reshape(t.size,1),(1,r.shape[1]))
    r_new=zeros((new_t.size,r.shape[1]))
    for i in range(r.shape[1]):
        #print i
        t_array[:,i]-=time_zero[i]-shift
        k=splrep(t_array[:,i],r[:,i],re[:,i],s=s)
        r_new[:,i]=splev(new_t,k)
    return r_new

def legend_format(l):
    return [str(i/1000.)+ ' ps' for i in l]

def savitzky_golay(y, window_size, order, deriv=0):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
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
        the swww.http.com//scipy-central.org/item/37/1/generating-confidence-intervals-via-model-comparsionsmoothed signal (or it's n-th derivative).
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




import ransac

def find_time_zero(d, value, method='abs', polydeg=3, *args):
    """
    Fits the dispersion of the timezero-point with a polynom of given
    degree. Uses the ransac algorithm to fit a approximation won by given
    method. The default method is to find the first point where the absolute 
    value is above a given limit.
    """
    
    t=d.t
    w=d.wl
    dat=d.data

    
    if method == 'abs':
        tn=t[np.argmin((np.abs(dat) < value),0)]
    
    ransac_data= np.column_stack((w,tn))    
    ransac_model = PoloynomialLeastSquaredModel(polydeg)
    m = ransac.ransac(ransac_data, ransac_model, tn.size*0.2, 500, 0.1, tn.size*0.6)
    
    return np.poly1d(m)(w), m
        
import numpy as np

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
    
    
def efa(dat, n, reverse=False):
    """ Doing evolving factor analyis.
    """
    if reverse: 
        data=dat[::-1, :]
    else:
        data=dat
        
    out=np.zeros((data.shape[0], n))
    for i in range(1, data.shape[0]):
        sv = np.linalg.svd(data[:i, :])[1]        
        out[i,:min(i, n)]=sv[:min(i, n)]
        
    return out
    
#from sklearn.decomposition import PCA
def moving_efa(dat, n, ncols, method='svd'):
    out=np.zeros((dat.shape[0], n))
    p=PCA()
    for i in range(0, dat.shape[0]-n):
        if method=='svd':
            sv = np.linalg.svd(dat[i:i+ncols,:])[1][:n]
        elif method=='pca':
            p = p.fit(dat[i:i+ncols,:])
            sv = p.explained_variance_ratio_[:n]
        out[i,:] = sv
    return out

from scipy.optimize import nnls 


#from sklearn.linear_model import Ridge
#r=Ridge(1.0, False)


def als(dat, n=5):   
  
    u, s, v = np.linalg.svd(dat)
    u0=u[:n]
    v0=v.T[:n]    
    
    res = np.linalg.lstsq(u0.T,dat)[1].sum()
    
    for i in range(2000):
        if i % 2 == 0:
            v0 = do_nnls(u0.T, dat)
            #v0 = v0/v0.max()
        else:
            
            u0, res_n, t , t = np.linalg.lstsq(v0.T, dat.T)
            ##r.fit(dat.T, v0.T)
            #u0 = r.coef_[:]
            res_n = res_n.sum()
            print res_n
            if res-res_n < 0.01: 
                break
                #
            else:
                res = res_n.sum()            
    return u0.T, v0.T


#import mls
def do_nnls(A,b):
    n = b.shape[1]
    out = np.zeros((A.shape[1], n))
    for i in xrange(n):              
        #mls.bounded_lsq(A.T, b[:,i], np.zeros((A.shape[1],1)), np.ones((A.shape[1],1))).shape
        out[:,i] =  nnls(A, b[:,i])[0]
    return out
    


    
if __name__=='__main__':
    #a = np.loadtxt('alcor_py2_ex400.txt')
    a=np.loadtxt('..\\altpfcpy2_620.dat')
    t = a[1:,0]
    wl = a[0,1:]
    a = a[1:, 1:]
    
    u, v = als(a[:,:].T)
    
    figure(0)
#    clf()
    subplot(211)
    plot(wl,u)
    subplot(212)
    plot(t[:],v)
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