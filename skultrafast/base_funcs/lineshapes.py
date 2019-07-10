# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:17:07 2013

@author: Tillsten
"""
import numpy as np
from scipy.special import erf


def lorentz(x, A, w, xc):
    return A / (1 + ((x - xc) / w)**2)


def gaussian(x, A, w, xc):
    return A * np.exp(((x - xc) / w)**2)


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
    return amp * 0.5 * (1 + erf((x - center) / sigma / np.sqrt(2)))
