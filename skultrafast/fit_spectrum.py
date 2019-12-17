# -*- coding: utf-8 -*-
"""
Module to fit the whole spektrum by peak functions.
"""

from __future__ import print_function
import scipy.optimize as opt
from scipy.special import wofz
from . import unit_conversions
from . import dv
import numpy as np
import lmfit
pi = np.pi


def voigt(x, A, mu, sig, gamma=0.1):
    w = wofz(((x-mu) + 1j*gamma) * 2**-0.5 / sig)
    return A * w.real * (2 * pi)**-0.5 / sig


def lorentz_peaks(x, A, x0, w):
    A, x0, w = map(np.asarray, [A, x0, w])
    return A[:, None] / (1 + ((x[None, :] - x0[:, None]) / w[:, None])**2)


def gauss_peaks(x, A, x0, w):
    A, x0, w = map(np.asarray, [A, x0, w])
    return A[:, None] * np.exp(-0.5 * ((x[None, :] - x0[:, None]) / w[:, None])**2)


def voigt_peaks(x, A, x0, w):
    out = np.zeros((A.size, x.size))
    for i in range(A.size):
        out[i, :] = voigt(x, A[i], x0[i], w[i])
    return out


def fit_spectrum(x,
                 y,
                 start_peaks_list,
                 yerr=None,
                 peak_func=lorentz_peaks,
                 amp_penalty=0.01,
                 amp_bounds=(-.6, .4),
                 wmin=2,
                 wmax=10,
                 add_const=False):
    """
    Fits multiple peaks to mulitple spektra, the position and width of
    each peak is the same for all spectra, only the amplitude is
    allowed to differ.

    Parameters
    ----------
    x: (n)-ndarray
       The x-values to fit, e.g. wavelengths or wavenumbers.
    y: (n, m)-ndarray
       The y-values to fit.
    start_peak_list: list
       A list containing (x0, amp, w) tuples. Used as starting values.
    yerr: (n, m)-ndarray
       The errors of the data. Default None.
    peaks_func: function, optional
        Function which calculates the peaks. Has the following signature:
        func(x, A_arr, x0_arr, w_arr), defaults to lorentz_peaks.
    amp_penalty: float, optional
       Regulazition parameter for the amplitudes. Defaults to 0.001.
    amp_bounds: (float, float)-tuple, optional
       Min and max bounds for the amplitude.
    wmax: float, optional
       Upper bound for the width parameter.
    wmin: float, optional
       Lower bound for the width parameter.
    add_const: bool
       Weather to add an const background.
    """
    y = np.atleast_2d(y)
    n = y.shape[0]
    paras = lmfit.Parameters()
    for i, (x0, A, w) in enumerate(start_peaks_list):
        si = str(i)
        for j in range(y.shape[0]):
            if A < 0:
                paras.add('Amp_' + si + str(j), A, max=0, min=amp_bounds[0])
            else:
                paras.add('Amp_' + si + str(j), A, max=amp_bounds[1], min=0)
        paras.add('Angle_' + si, 54.2, max=90, min=0)
        paras.add('x0_' + si, x0), print(x0)
        paras.add('width_' + si, w, min=wmin, max=wmax)
    p = paras
    #print(p)
    x0 = np.array([i.value for i in p.values()])

    #up_bounds = np.array([i.max for i in p.values()])
    #min_bounds = np.array([i.min for i in p.values()])

    def residuals(p, x, y, peak_func):
        fit = np.array([i.value for i in p.values()]).reshape((3 + n, -1), order='f')
        #fit = p.reshape((3+n, -1), order='f')
        base_peak = peak_func(x, np.ones_like(fit[0, :]), *fit[[-2, -1], :])

        dichro = unit_conversions.angle2dichro(fit[-3, :])

        resi = []
        for i in range(n):

            fp = base_peak * fit[[i], :].T
            fs = fp / dichro[:, None]
            sum_fs = fs.sum(0)
            sum_fp = fp.sum(0)

            if y is None:
                resi.append(np.hstack((fp, fs)))
            else:
                resi.append(
                    np.hstack((y[i, :] - np.hstack(
                        (sum_fp, sum_fs)), fit[i, :] * amp_penalty)).ravel())

        if y is None:
            return np.array(resi)
        else:
            if yerr is None:
                return np.array(resi).ravel()
            else:
                return np.array(resi / yerr).ravel()

    print(x.shape)
    mini = lmfit.Minimizer(residuals, paras, fcn_args=(x, y, peak_func))
    result = mini.leastsq()
    #result = mini.scalar_minimize('BFGS')
    #result = opt.least_squares(residuals, x0, bounds=(min_bounds, up_bounds),
    #                           args=(x, y, peak_func), jac='3-point')
    #for k, i in enumerate(paras):
    #    paras[i].value = result.x[k]
    return result, residuals, mini  # (up_bounds,  min_bounds)


import astropy.stats as st


def bin_every_n(x, start_idx, n=10, reduction_func=lambda x: np.mean(x, 0)):
    out = []
    if x.ndim == 1:
        x = x[:, None]
    for i in range(start_idx, x.shape[0], n):
        end_idx = min(i + n, x.shape[0])
        out.append(st.sigma_clip(x[i:end_idx, :], sigma=2.5, maxiters=1, axis=0).mean(0))
    return np.array(out)