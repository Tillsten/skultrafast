# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 16:06:14 2014

@author: Tillsten
"""

from __future__ import print_function
import numpy as np
from sklearn import linear_model as lm
from skultrafast import dv, zero_finding
from skultrafast.base_functions_np import _fold_exp, _coh_gaussian

def _make_base(tup, taus, w=0.1, add_coh=True, add_const=False, norm=False):
    if add_const:
        taus = np.hstack((taus, 10000))
    out = _fold_exp(tup.t.T[:, None], w, 0, taus[None, :]).squeeze()
    if add_const:
        print(out.shape)
        out[:, -1] *= 1000
    if add_coh:
        out = np.hstack((out, _coh_gaussian(tup.t.T[:, None], w, 0).squeeze())) * 10
    if norm:
        out = out / np.abs(out).max(0)

    return out.squeeze()


def start_ltm(tup, taus, w=0.1,  add_coh=False, use_cv=False,
              add_const=False, verbose=False, **kwargs):
    """
    Parameter
    ---------

    tup:
        namedtuple/class with t, wl and data as attributes.
        t array shape N, wl array shape M, data array (N, M)

    """
    X = _make_base(tup, taus, w=w,
                   add_const=add_const,
                   add_coh=add_coh)
    if not use_cv:
        mod = lm.ElasticNet(**kwargs)

    else:
        mod = lm.ElasticNetCV(**kwargs)
    mod.fit_intercept = 1
    mod.warm_start = 1

    coefs = np.empty((X.shape[1], tup.data.shape[1]))
    fit = np.empty_like(tup.data)
    alphas = np.empty(tup.data.shape[1])

    for i in range(tup.data.shape[1]):
        if verbose:
            print(i, 'ha', end=';')
        mod.fit(X, tup.data[:, i])
        coefs[:, i] = mod.coef_.copy()
        fit[:, i] = mod.predict(X)
        if hasattr(mod, 'alpha_'):
            alphas[i] = mod.alpha_
    return mod, coefs, fit, alphas


def start_ltm_multi(tup, taus, w=0.1, alpha=0.001, **kwargs):
    X = _make_base(tup, taus, w=w)
    mod = lm.MultiTaskElasticNet(alpha=alpha,  **kwargs)
    mod.max_iter = 5e4
    mod.verbose = 0
    mod.fit_intercept = 0
    mod.normalize = 1
    mod.fit(X, tup.data)

    fit = mod.predict(X)
    coefs = mod.coef_
    return mod, coefs, fit, None