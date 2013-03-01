# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 18:06:20 2013

@author: Tillsten
"""
import numpy as np
import sklearn.linear_model as lm
from skultrafast import dv
from skultrafast.fitter import _fold_exp, _coh_gaussian
import scipy.signal as sig
import matplotlib.pyplot as plt

def _make_base(tup, taus, w=0.8, add_coh=False):
     return np.exp(-tup.t[:, None] / taus[None, :])
#    out = _fold_exp(tup.t, w, 0, taus).T
#    if add_coh:
#        out = np.hstack((out, _coh_gaussian(tup.t, w, 0)))
#    return out

def lifetime_map(tup, taus, method='elastic', **kwargs):
    wl, t, dat = tup.wl, tup.t, tup.data
    X = _make_base(tup, taus)
    res = np.empty_like(dat)
    coefs = np.empty((taus.size, dat.shape[1]))
    if method == 'elastic':
        mod = lm.ElasticNet(**kwargs)
    elif method == 'elasticCV':
        mod = lm.ElasticNetCV(**kwargs)
    elif method == 'lasso':
        mod = lm.Lasso(**kwargs)
    elif method == 'lassoCV':
        mod = lm.LassoCV(**kwargs)
    #for i in range(dat.shape[1]):
        #mod.fit(X, y[:, i])
    mod.fit(X, dat)
    return mod.coef_, mod.predict(X)
    
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
    

taus = np.logspace(-1, 4, 200)
#u, o = dv.binner(200, f.wl, f.data)
#tup = dv.tup(o, f.t[:], u[11:,:])
k, nwl = dv.binner(100, f.wl, m.data)
k /= k[-2:-1,:]
tup = dv.tup(nwl, m.t[35:], k[35:,:])

s = lm.Lasso(fit_intercept=True, max_iter=1e5)     
X = _make_base(tup, taus)  
s.warm_start = True 
for i, alpha in enumerate(np.linspace(0.001, 0.03, 3)):    
    s.alpha = alpha
    s.fit(X, tup.data)
    subplot(3,1, i+1)
    title(str(alpha))
    normed = s.coef_/np.abs(s.coef_).max(0)[None,:]
    pcolormesh(tup.wl, taus, s.coef_)
    yscale('log')
    #clim(-abs(s.coef_).max(), abs(s.coef_).max())
    set_cmap(cm.RdBu)

