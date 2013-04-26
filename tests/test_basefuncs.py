# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:34:15 2013

@author: Tillsten
"""

import nose
from skultrafast.base_functions import fast_erfc, calc_gaussian_fold, \
      _fold_exp, _coh_gaussian, _exp
      
import skultrafast.fitter_cython
from numpy.testing import assert_array_almost_equal
import numpy as np


def test_fast_erfc():
    from scipy.special import erfc as erfc_s
    x = np.linspace(-3, 3, 200)
    y = np.array([fast_erfc(i) for i  in x])
    assert_array_almost_equal(erfc_s(x), y, 3)


def test_fold_exp():
    taus = np.array([1., 20., 30.])
    t_array = np.subtract.outer(np.linspace(-1, 50, 300),
                                np.linspace(3, 3, 400))
    w = 0.1
    dt = np.diff(t_array, 1, 0)[0, 0]

    y = _fold_exp(t_array, w, 0, taus)
    return y

def test_exp():
    taus = np.array([1., 20., 30.])
    t_array = np.subtract.outer(np.linspace(0, 50, 300),
                                np.linspace(0, 0, 400))
    w = 0.1
    y = _exp(t_array, w, 0, taus)    
    np.testing.assert_almost_equal(np.exp(-t_array[:, 0]), y[:, 0, 0])
    

def test_folded_eq_exp():
    """
    For t>>w exp==folded exp
    """
    taus = np.array([1., 20., 30.])
    t_array = np.subtract.outer(np.linspace(10, 50, 300),
                                np.linspace(3, 3, 400))
    w = 0.1
    y = _fold_exp(t_array, w, 0, taus)
    y2 = _fold_exp(t_array, w, 0, taus)    
    np.testing.assert_array_almost_equal(y, ynp.exp())

if __name__ == '__main__':
     import matplotlib.pyplot as plt

     a = test_fold_exp()
#
#     plt.plot(a[:, 0, :])
#     plt.show()
     
     b = test_exp()
     print a.shape
     print b.shape
     
     plt.plot(b[:, 9, :], lw=2)
     plt.plot(a[:, 9, :], lw=2)
     plt.show()

    # nose.run()