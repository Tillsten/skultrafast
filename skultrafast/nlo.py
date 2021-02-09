"""
Module containing helpers for small calculation involing nonlinear optics
"""
# %%
from skultrafast.unit_conversions import c
from scipy.optimize import minimize_scalar
import numpy as np

def tl_pulse_from_nm(center_wl: float, fhwm: float, shape: str = 'gauss') -> float:
    """
    Calculates the transformlimted pulselength in fs from given center
    wavelength and fwhm in nanometers.

    Parameters
    ----------
    center_wl : float 
    fhwm : float 
    shape : str,
       optional, by default 'gauss'

    Returns
    -------
    float 
       
    """
    if shape == 'gauss':
        tbw = 0.44
    elif shape == 'sech':
        tbw = 0.315
    
    return tbw / (c*1e9 * fhwm / center_wl**2)*1e15


def pulse_length(t_in, phi_2):
    f = 4*np.log(2)*phi_2 / t_in**2
    t_out = t_in * np.sqrt(1+f**2)
    return t_out


def dispersion(t_in, t_out):
    """
    Estimates the amount of dispersion assuming form the pulse length a
    transform limited input pulse

    Parameters
    ----------
    t_in : float
        [description]
    t_out : float
        [description]

    Returns
    -------
    [type]
        [description]
    """
    f = minimize_scalar(lambda x: (pulse_length(t_in, x) - t_out)**2)
    return f.x


def dist(d, alpha=10):
    a = np.deg2rad(alpha)
    lot = np.cos(a/2)*d
    dist = np.sin(a)*d
    return dist*2
tl = tl_pulse_from_nm(765, 20)
dispersion(tl, 120)/54
25/dist(5)

import matplotlib.pyplot as plt

plt.figure(dpi=200)
d = np.linspace(4, 20, 100)
plt.plot(d, 2*np.floor(25.4/dist(d)))
plt.setp(plt.gca(), xlabel='distance mirrors (mm)', ylabel='max. bounces')
plt.annotate('Bounces at 10° AOI', (10, 25), fontsize='large')
dist(280)


# %%
a = np.arange(1,7)
for i in range(4):
    a = np.dstack((a, a))
a.shape
# %%
6**4
# %%
import itertools
# %%
a = itertools.product(range(1, 7), repeat=3)
ar = np.array(list(a))
ar = ar.sum(1)
#plt.hist(ar.sum(1), bins=np.arange(3, 20),  histtype='step', density=True)
plt.step(np.arange(ar.max()+1), np.bincount(ar)/len(ar))
a = itertools.product(range(1, 7), repeat=4)
ar = np.array(list(a))
ar = ar.sum(1)-ar.min()
#plt.hist(ar.sum(1), bins=np.arange(3, 25), histtype='step')
plt.step(np.arange(ar.max()+1), np.bincount(ar)/len(ar))

# %%
itertools.co