# -*- coding: utf-8 -*-
"""
Module to import the base functions from.
"""
from __future__ import print_function

try:
    from skultrafast.base_funcs.base_functions_cl import (_fold_exp,
                                                          _coh_gaussian,
                                                          _fold_exp_and_coh)
    print("pyopencl found, using cl-basefunctions")
except ImportError:
    try:
        from skultrafast.base_funcs.base_functions_numba import (_fold_exp,
                                                                 _fold_exp_and_coh,
                                                                 _coh_gaussian)
        print("pyopencl not found, using numba-basefunctions")
    except ImportError:
        from skultrafast.base_funcs.base_functions_np import(_fold_exp,
                                                             _fold_exp_and_coh,
                                                             _coh_gaussian)
        print("pyopencl and numba not found, using pure numpy-basefunctions")
