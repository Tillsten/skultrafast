# -*- coding: utf-8 -*-
"""
Created on Tue Jul 01 19:51:56 2014

@author: Tillsten
"""
import numpy as np
import skultrafast.dv as dv
from scipy.stats import trim_mean, linregress


def scan_correction(dn, tidx):
    for j in [0, 1]:
        null_spek =  dn[tidx:, :, j, 0].mean(0)
        null_std = dn[tidx:, :, j, 0].std(0)
        for i in range(0, dn.shape[-1], 2):
            spec = dn[tidx:, :, j, i].mean(0)
            c = np.linalg.lstsq(spec[:, None], null_spek[:, None])            
            dn[:, :, j, i] *= c[0][0]

        null_spek =  dn[tidx:, :, j, 1].mean(0)
        for i in range(1, dn.shape[-1], 2):
            spec = dn[tidx:, :, j, i].mean(0)
            c = np.linalg.lstsq(spec[:, None], null_spek[:, None])
            dn[:, :, j, i] *= c[0][0]
    return dn

def load(fname, recalc_wl=True):
    f = np.load(fname)
    t = f['t']/1000.
    wl = f['wl']
    if recalc_wl:
        for i in range(wl.shape[1]):        
            wl[:, i] = (np.arange(32)-16)*8.4 + wl[16, i]
    data = -f['data']
    return t, 1e7/wl, data

def calc_fac(a, b, tidx):
    a =  a[tidx:, :].mean(0)[:, None]
    b =  b[tidx:, :].mean(0)[:, None]
    c = np.linalg.lstsq(b.ravel()[:, None], a.ravel()[:, None])
    return c[0][0]

def shift_linear_part(p, steps, t):
    "Shift the linear part of the array by steps"
    lp = dv.fi(t, -1), dv.fi(t, 3)
    p = p.copy()
    p[lp[0]+steps:lp[1], ...] = p[lp[0]:lp[1]-steps, ...]
    return p

import statsmodels.api as sm



def robust_mean_back(d, n=10):
    d = d.copy()
    mean_backs = d[:n, ...].mean(0)
    mean_backs_std = d[:n, ...].std(0)
    return np.average(mean_backs, -1, 1/mean_backs_std**2).mean(-1)


def back_correction(d, n=10, use_robust=True):
    print d.shape
    d = d.copy()
    mean_back = d[:n, ...].mean(0).mean(-1).mean(-1)
    scan_means = d[:n, ...].mean(0)

    out0 = []
    out1 = []
    if use_robust:
        for i in range(d.shape[-1]):
            rlm_model = sm.RLM(scan_means[:, 0, i], mean_back[:, None],
                               M=sm.robust.norms.HuberT())
            rlm_results = rlm_model.fit()
            out0.append(rlm_results.params)

            rlm_model = sm.RLM(scan_means[:, 1, i], mean_back[:, None],
                               M=sm.robust.norms.HuberT())
            rlm_results = rlm_model.fit()
            out1.append(rlm_results.params)

        c1 = np.array(out0).T
        c2 = np.array(out1).T
    else:
        c1 = np.linalg.lstsq(mean_back[:, None], scan_means[:, 0, : ])[0]
        c2 = np.linalg.lstsq(mean_back[:, None], scan_means[:, 1, : ])[0]


    print c1.shape

    d[..., 0, :] -= mean_back[:, None].dot(c1)[None, :,  :]
    d[..., 1, :] -= mean_back[:, None].dot(c2)[None, :,  :]
    return d, mean_back[...]


import scipy.signal as sig
import scipy.ndimage as nd
def data_preparation(wl, t, d, wiener=3, trunc_back=0.05, trunc_scans=0, start_det0_is_para=True,
                     do_scan_correction=True, do_iso_correction=True, plot=1, n=10):
    d = d.copy()
    #d[..., 0, :]= shift_linear_part(d[..., 0, :])
    #d[..., 1, :] = shift_linear_part(d[..., 1, :], 1, t)

    if wiener == 'svd':
        for i in range(d.shape[-1]):
            d[:, :, 0, i] = dv.svd_filter(d[:, :, 0, i], 5)
            d[:, :, 1, i] = dv.svd_filter(d[:, :, 1, i], 5)
    elif wiener > 1:
        d = sig.wiener(d, (wiener, 1, 1, 1))
    elif wiener < 0:
        d = nd.uniform_filter1d(d, -wiener, 0, mode='nearest')
    


    #d, back0 = back_correction(d, use_robust=1)
    #back1 = back0
    fi = lambda x, ax=0: trim_mean(x, trunc_back, ax)
    back0 = fi(d[:n, ..., 0, :], ax=0)
    back1 = fi(d[:n, ..., 1, :], ax=0)
    back = 0.5*(back0+back1).mean(-1)
    d[..., 0, :] -= back.reshape(1, 32, -1)
    d[..., 1, :] -= back.reshape(1, 32, -1)
    
    if do_scan_correction:
        d = scan_correction(d, dv.fi(t, 0))


    #gr -> vert -> parallel zum 0. scan
    fi = lambda x, ax=-1: trim_mean(x, trunc_scans,  ax)
    fi = lambda x: dv.trimmed_mean(x, ratio=trunc_scans)[0]
    #fi = lambda x, ax=-1: np.median(x, ax)
    if start_det0_is_para:
        para_0 = fi(d[..., 0, ::2])
        senk_0 = fi(d[..., 0, 1::2])
        para_1 = fi(d[..., 1, 1::2])
        senk_1 = fi(d[..., 1, 0::2])
    else:
        para_0 = fi(d[..., 0, 1::2])
        senk_0 = fi(d[..., 0, 0::2])
        para_1 = fi(d[..., 1, 0::2])
        senk_1 = fi(d[..., 1, 1::2])


    iso_0 = (para_0 + 2*senk_0) / 3.
    iso_1 = (para_1 + 2*senk_1) / 3.

    if do_iso_correction:
        iso_factor = calc_fac(iso_0, iso_1, dv.fi(t, 1))
    else:
        iso_factor = 1

    senk = 0.5*senk_0 + 0.5 * (iso_factor*senk_1)
    senk -= senk[:10, :].mean(0)
    para = 0.5*para_0 + 0.5 * (iso_factor*para_1)
    para -= para[:10, :].mean(0)
    iso = (2*senk + para)/3
    if plot:
        import matplotlib.pyplot as plt
        from plot_helpers import lbl_spec, mean_spec
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.plot(wl, iso[:10, :].mean(0))
        plt.plot(wl, back0)
        plt.plot(wl, back1)
        lbl_spec()
        plt.legend(['iso_rest', 'back0', 'back1'])
        plt.subplot(122)
        mean_spec(wl, t, [iso_0, iso_factor*iso_1], (1, 100))
        mean_spec(wl, t, [para, senk], (1, 100), color='r')
        plt.legend(['iso_0', 'iso_1 * %.2f'%iso_factor])

    return iso, para, senk
