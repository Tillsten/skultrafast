"""
This module provides a class for calculating the pseudo-Zernike polynomials for
2DIRT spectra.

It is based on "A new method based on pseudo-Zernike polynomials to analyze and
extract dynamical and spectral information from the 2DIR spectra" by Gurung  &
Kuroda.
"""
# %%

# %%
import numpy as np

from skultrafast.pseudo_zernike.poly import Polybase
from skultrafast.twoD_dataset import TwoDim
import matplotlib.pyplot as plt

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Parameters
    ----------
    a : np.ndarray
        The first vector
    b : np.ndarray
        The second vector

    Returns
    -------
    similarity : float
        The cosine similarity
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def total_pzs(ds: TwoDim, n: int) -> np.ndarray:
    """
    Calculate the pseudo-Zernike polynomials for the 2DIR spectra.

    Parameters
    ----------
    ds : TwoDim
        The 2DIR spectra
    n : int
        The maximum order of the polynomials

    Returns
    -------
    pzs : np.ndarray
        The pseudo-Zernike polynomials
    """
    pb = Polybase(ds.pump_wn, ds.probe_wn, n)
    base = pb.make_base()


