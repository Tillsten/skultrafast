"""
This module contains functions to covert between units.
"""
import numpy as np
from scipy.constants import physical_constants, c

c_cm = c * 100
names = dict(cm="wavenumbers in 1/cm",
             fs="period in femotoseconds",
             nm="wavelength in nanometers",
             eV="energy in electron Volt",
             THz="frequency in THz",
             dichro="Dichroic ratio (para/perp)",
             angle="relative angle between transition dipole moments in degrees",
             aniso="Anisotropy (para-perp)/(para+2*perp)",
             kcal="energy in kcal/mol")


def make_doc(func):
    a, b = str.split(func.__name__, '2')
    func.__doc__ = ('%s to %s' % (names[a], names[b])).capitalize()
    return func


@make_doc
def fs2cm(t):
    return 1 / (t*1e-15*c_cm)


@make_doc
def cm2fs(cm):
    return 1e15 / (cm*c_cm)


@make_doc
def nm2cm(nm):
    return 1e7 / nm


@make_doc
def cm2nm(cm):
    return 1e7 / cm


@make_doc
def cm2eV(cm):
    eV_m = physical_constants['electron volt-inverse meter relationship'][0]
    eV_cm = eV_m / 100
    return cm / eV_cm


@make_doc
def eV2cm(eV):
    eV_m = physical_constants['electron volt-inverse meter relationship'][0]
    eV_cm = eV_m / 100
    return eV * eV_cm


@make_doc
def cm2THz(cm):
    return 1 / fs2cm(cm) / 1e-3


@make_doc
def THz2cm(THz):
    return cm2fs(1e3 / THz)


@make_doc
def dichro2angle(d):
    return np.arccos(np.sqrt((2*d - 1) / (d+2))) / np.pi * 180


@make_doc
def angle2dichro(deg):
    rad = np.deg2rad(deg)
    return (1 + 2 * np.cos(rad)**2) / (2 - np.cos(rad)**2)


@make_doc
def angle2aniso(deg):
    ang = np.deg2rad(deg)
    return 2 / 5 * (3 * np.cos(ang)**2 - 1) / 2


@make_doc
def aniso2angle(r):
    return np.arccos(np.sqrt((r*10/2 + 1) / 3)) / np.pi * 180


@make_doc
def cm2kcal(cm):
    return cm * 2.859e-3


@make_doc
def kcal2cm(kcal):
    return kcal / 2.859e-3
