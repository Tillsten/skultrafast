# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:34:15 2013

@author: Tillsten
"""


from skultrafast.base_funcs.base_functions_numba import fast_erfc, _fold_exp, _exp

import skultrafast.base_funcs.base_functions_np as bnp
try:
    import skultrafast.base_funcs.base_functions_cl as bcl
except ImportError:
    print('Warning, pyopencl was not found. OpenCL backend ist not tested')
    bcl = bnp
import skultrafast.base_funcs.base_functions_numba as bnb

from numpy.testing import assert_array_almost_equal
import numpy as np
import pytest

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


def test_folded_equals_exp():
    """
    For t>>w exp==folded exp
    """
    taus = np.array([1., 20., 30.])
    t_array = np.subtract.outer(np.linspace(40, 50, 300),
                                np.linspace(3, 3, 400))
    w = 0.1
    y = _fold_exp(t_array, w, 0, taus)
    y2 = _fold_exp(t_array, w, 0, taus)
    exp_y = np.exp(-t_array[ :, :, None]/taus[ None, None,:])
    np.testing.assert_array_almost_equal(y, exp_y)


def test_compare_fold_funcs():
    taus = np.array([1., 20., 30.])
    t_array = np.subtract.outer(np.linspace(-2, 50, 300),
                                np.linspace(-1, 3, 400))
    w = 0.1
    y1 = bnp._fold_exp(t_array, w, 0, taus)
    y2 = bcl._fold_exp(t_array, w, 0, taus)
    np.testing.assert_array_almost_equal(y1, y2, 4)

    y3 = bnb._fold_exp(t_array, w, 0, taus)
    np.testing.assert_array_almost_equal(y1, y3, 3)

@pytest.mark.xfail
def test_compare_coh_funcs():
    t_array = np.subtract.outer(np.linspace(-4, 4, 300),
                                np.linspace(3, 3, 400))
    w = 0.1
    y1 = bnb._coh_gaussian(t_array,  w, 0.)
    y2 = bcl._coh_gaussian(t_array,  w, 0.)
    np.testing.assert_array_almost_equal(y1, y2, 4)

if __name__ == '__main__':
    test_compare_coh_funcs()

#     import matplotlib.pyplot as plt

#     a = test_fold_exp()
##
##     plt.plot(a[:, 0, :])
##     plt.show()
#
#     b = test_exp()
#     print a.shape5
#     plt.plot(b[:, 9, :], lw=2)
#     plt.plot(a[:, 9, :], lw=2)
#     plt.show()

   # nose.run()
