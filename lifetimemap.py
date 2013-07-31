# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 18:06:20 2013

@author: Tillsten
"""
import numpy as np
import sklearn.linear_model as lm
from skultrafast import dv, zero_finding
from skultrafast.fitter import _fold_exp, _coh_gaussian
import scipy.signal as sig
import matplotlib.pyplot as plt

def _make_base(tup, taus, w=0.8, add_coh=False):
     return np.exp(-tup.t[:, None] / taus[None, :])
#    out = _fold_exp(tup.t, w, 0, taus).T
#    if add_coh:
#        out = np.hstack((out, _coh_gaussian(tup.t, w, 0)))
#    return out
     
def make_ltm(fitter, lowest_time=0.25, max_bins=200, 
             n_taus=60, taus_lim=(-1, 2.5)):
    wl, t, dat = fitter.wl, fitter.t, fitter.data
    if fitter.data.shape[1]>max_bins or max_bins==0:
        dat, wl = dv.binner(max_bins, wl, dat)    
    tidx = dv.fi(t, lowest_time)
    t, dat = t[tidx:], dat[tidx:, :]
    taus = np.logspace(taus_lim[0], taus_lim[1], n_taus)
    coefs, fit, alpha = ltm(dv.tup(wl, t, dat), taus, 'lassoCV', n_alphas=10, 
                            max_iter=1e4)
    print alpha
    return coefs, fit, taus, dv.tup(wl, t, dat)
    

def ltm(tup, taus, method='lassoCV', **kwargs):
    wl, t, dat = tup.wl, tup.t, tup.data
    X = _make_base(tup, taus)
    print X.shape
    res = np.empty_like(dat)
    #coefs = np.empty((taus.size, dat.shape[1]))
    if method == 'elastic':
        mod = lm.ElasticNet(**kwargs)
    elif method == 'elasticCV':
        mod = lm.ElasticNetCV(**kwargs)
    elif method == 'lasso':
        mod = lm.Lasso(**kwargs)
    elif method == 'lassoCV':
        mod = lm.LassoCV(**kwargs)
    elif method == 'larsCV':
        mod = lm.LarsCV(**kwargs)
    #for i in range(dat.shape[1]):
        #mod.fit(X, y[:, i])
    print dat
    mod.fit(X, dat)
    return mod.coef_.copy(), mod.predict(X), mod.alpha_
    
def annotate(taus, data):
    idx = np.unique(np.hstack((sig.argrelmax(data, 0, 5), sig.argrelmin(data, 0, 5))))
    points = taus[idx]
    plt.vlines(points, np.zeros_like(points), data[idx])
    for i in idx:
        plt.text(taus[i], data[i] +np.sign(data[i])*0.3,
                 '{0: .2f}'.format(taus[i]), ha='center')
#plot(taus, s.coef_[14, :])
#annotate(taus, s.coef_[14, :])    
#xscale('log')
def contiuenes_reg(X, ):
    pass
#f = zero_finding.interpol(g, g.tn)
#c, f, taus, tup = make_ltm(f)
##
#taus = np.logspace(-1, 2.5, 100)
###u, o = dv.binner(200, f.wl, f.data)
###tup = dv.tup(o, f.t[:], u[11:,:])
##k, nwl = dv.binner(100, f.wl, m.data)
##k /= k[-2:-1,:]
#tup = dv.tup(f.wl, f.t, f.data)
#
#s = lm.ElasticNet(fit_intercept=True, max_iter=1e5)     
#s.l1_ratio = .99
#s.alpha = 0.01
#X = _make_base(tup, taus)  
##s.warm_start = True 
##for i, alpha in enumerate(np.linspace(0.001, 0.03, 3)):    
##    #s.alpha = alpha
#s.fit(X, tup.data)
##    subplot(3,1, i+1)
##    title(str(alpha))
##    normed = s.coef_/np.abs(s.coef_).max(0)[None,:]
#pcolormesh(tup.wl, taus, s.coef_.T)
#yscale('log')
#set_cmap(cm.Spectral)
#colorbar()
#clim(-abs(s.coef_).max(), abs(s.coef_).max())
#
##
#autoscale(1, 'both', 'tight')