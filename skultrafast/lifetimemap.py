# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from typing import Tuple, List, Iterable
from sklearn import linear_model as lm
from skultrafast.base_funcs.base_functions_np import _fold_exp, _coh_gaussian


def _make_base(tup, taus, w=0.1, add_coh=True, add_const=False, norm=False):
    if add_const:
        taus = np.hstack((taus, 10000))
    out = _fold_exp(tup.t.T[:, None], w, 0, taus[None, :]).squeeze()
    if add_const:
        print(out.shape)
        out[:, -1] *= 1000
    if add_coh:
        out = np.hstack(
            (out, _coh_gaussian(tup.t.T[:, None], w, 0).squeeze())) * 10
    if norm:
        out = out / np.abs(out).max(0)

    return out.squeeze()


def start_ltm(tup,
              taus,
              w=0.1,
              add_coh=False,
              use_cv=False,
              add_const=False,
              verbose=False,
              **kwargs):
    """Calculate the lifetime density map for given data.

    Parameters
    ----------
    tup : datatuple
        tuple with wl, t, data
    taus : list of floats
        Used to build the basis vectors.
    w : float, optional
        Used sigma for calculating the , by default 0.1.
    add_coh : bool, optional
        If true, coherent contributions are added to the basis.
        By default False.
    use_cv : bool, optional
        Whether to use cross-validation, by default False
    add_const : bool, optional
        Whether to add an explict constant, by default False
    verbose : bool, optional
        Wheater to be verobse, by default False

    Returns
    -------
    tuple of (linear_model, coefs, fit, alphas)
        The linear model is the used sklearn model. Coefs is the arrary
        of the coefficents, fit contains the resulting fit and alphas
        is an array of the applied alpha value when using cv.
    """

    X = _make_base(tup, taus, w=w, add_const=add_const, add_coh=add_coh)
    if not use_cv:
        mod = lm.ElasticNet(**kwargs, l1_ratio=0.98)

    else:
        mod = lm.ElasticNetCV(**kwargs, l1_ratio=0.98)

    mod.fit_intercept = not add_const
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
    mod = lm.MultiTaskElasticNet(alpha=alpha, **kwargs)
    mod.max_iter = 5e4
    mod.verbose = 0
    mod.fit_intercept = 0
    mod.normalize = 1
    mod.fit(X, tup.data)

    fit = mod.predict(X)
    coefs = mod.coef_
    return mod, coefs, fit, None
