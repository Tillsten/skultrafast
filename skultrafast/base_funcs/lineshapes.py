# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:17:07 2013

@author: Tillsten
"""
import numpy as np
from scipy.special import erf


def lorentz(x, A, w, xc):
    return A / (1 + ((x-xc) / w)**2)


def gaussian(x, A, w, xc):
    return A * np.exp(((x-xc) / w)**2)


def gauss_step(x, amp: float, center: float, sigma: float):
    """Returns the stepfunction (erf-style) for given arguments.

    Parameters
    ----------
    x : array
        Independent variable
    amp : float
        Amplitude of the step
    center : float
        Position of the step 
    sigma : float
        Width of the step

    Returns
    -------
    array
        The step functions
    """
    return amp * 0.5 * (1 + erf((x-center) / sigma / np.sqrt(2)))


def gauss2d(pu, pr, A0, x_pr, x_pu, sigma_pu, sigma_pr, corr):
    pr = pr[:, None]
    pu = pu[None, :]
    c_pr = ((pr-x_pr) / sigma_pr)
    c_pu = ((pu-x_pu) / sigma_pu)
    y = -A0 * np.exp(-1 / (2 - 2 * corr**2) * (c_pr**2 - 2*corr*c_pr*c_pu + c_pu**2))
    return y


def single_gauss(pu, pr, A0, x01, sigma_pu, sigma_pr, corr):
    return gauss2d(pu, pr, A0, x01, x01, sigma_pu, sigma_pr, corr)


def two_gauss2D_shared(pu, pr, A0, x01, ah, sigma_pu, sigma_pr, corr):
    y = gauss2d(pu, pr, A0, x01, x01, sigma_pu, sigma_pr, corr)
    x12 = x01 - ah
    y += gauss2d(pu, pr, -A0, x12, x01, sigma_pu, sigma_pr, corr)
    return y


def two_gauss2D(pu, pr, A0, x01, ah, sigma_pu, sigma_pr, corr, A1, sigma_pu2, sigma_pr2,
                corr2):
    y = gauss2d(pu, pr, A0, x01, sigma_pu, sigma_pr, corr)
    x12 = x01 - ah
    y -= gauss2d(pu, pr, A1, x12, sigma_pu2, sigma_pr2, corr2)
    return y
