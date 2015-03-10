# -*- coding: utf-8 -*-
"""
Module to fit the whole spektrum by peak functions.
"""

from __future__ import print_function
import lmfit, dv, numpy as np
from scipy.special import wofz
pi = np.pi

def voigt(x, A, mu, sig, gamma=0.1):
      w = wofz (((x-mu)+1j*gamma)*2**-0.5/ sig)
      return A*w.real*(2*pi)**-0.5/ sig
    
def lorentz_peaks(x, A, x0, w):
    A, x0, w = map(np.asarray, [A, x0, w])
    return A[:, None]/(1+((x[None, :]-x0[:, None])/w[:, None])**2)

def voigt_peaks(x, A, x0, w):
    out = np.zeros((A.size, x.size))    
    for i in range(A.size):
        out[i, :] = voigt(x, A[i], x0[i], w[i])
    return out


    
start_peaks = [1670,  .1, 4,
               1665, -.3, 4,
               1655, -.2, 4,
               #1659, -.1, 4, 
               1645, .2, 4, 
               1630, -.2, 4,
               1620, .2, 4]


def fit_spectrum(x, y, start_peaks_list, 
                 peaks_func=lorentz_peaks,
                 amp_penalty=0.001,
                 amp_bounds=(-10, 10),
                 wmax=16):
    """
    Fits multiple peaks to mulitple spektra, the position and width of
    each peak is the same for all spectra, only the amplitude is
    allowed to differ.    
    
    Parameters
    ----------
    x: (n)-ndarray 
       The x-values to fit, e.g. wavelengths or wavenumbers.
    y: (n, m)-ndarray
       The y-vales to fit.
    start_peak_list: list
       A list containing (x0, amp, w) tuples. Used as starting values.       
    peaks_func: function, optional
        Function which calculates the peaks. Has the following signature:
        func(x, A_arr, x0_arr, w_arr), defaults to lorentz_peaks.
    amp_penalty: float, optional
       Regulazition parameter for the amplitudes. Defaults to 0.001.    
    amp_bounds: (float, float)-tuple, optional
       Min and max bounds for the amplitude.
    wmax: float, optional
       Upper bound for the width parameter.        
    """
    
    paras = lmfit.Parameters()
    for i, (x0, A, w) in enumerate(start_peaks_list):
        si = str(i)
        paras.add('Amp_' + si, A, max=amp_bounds[1], min=amp_bounds[0])
        paras.add('Angle_' + si, 52, max=90, min=0)
        paras.add('x0_'+ si, x0)
        paras.add('width_'+ si, w, min=0, max=wmax)


    def residuals(p, x, y, peak_func):    
        fit = np.array([i.value for i in p.values()]).reshape((4, -1), order='f')    
        fp = peak_func(x, *fit[[0,2,3], :])
        dichro = dv.angle_to_dichro(np.deg2rad(fit[1, :])) 
        fs = fp * dichro[:, None]
        sum_fp = fp.T.sum(1)
        sum_fs = fs.T.sum(1)
        if y is None:
            return fp, fs
        else: 
            return y-np.hstack((sum_fp, sum_fs))
            
            
    mini = lmfit.minimizer(paras, residuals)
    mini.leastsq()
    
start_peaks_list = zip(*[iter(start_peaks)]*3)
print(start_peaks_list)




    