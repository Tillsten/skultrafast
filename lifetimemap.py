# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 18:06:20 2013

@author: Tillsten
"""
import numpy as np
import sklearn.linear_model as lm
from skultrafast import dv, zero_finding
from skultrafast.fitter import _fold_exp, _coh_gaussian
from skultrafast.base_functions_np import _fold_exp
import scipy.signal as sig
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1
def _make_base(tup, taus, w=0.16, add_coh=False):
     print taus.shape
     out = _fold_exp(tup.t.T[:, None], w, 0, taus[None, :])
     print out.shape
#    if add_coh:
#        out = np.hstack((out, _coh_gaussian(tup.t, w, 0)))
     return out.squeeze()

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
    kwargs['max_iter'] = 20000
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
    elif method == 'larsCV':
        mod = lm.LarsCV(**kwargs)
    elif method == 'multiEN':
        mod = lm.MultiTaskElasticNet(**kwargs)
        mod.l1_ration = .99
    #for i in range(dat.shape[1]):
        #mod.fit(X, y[:, i])
    print dat
    mod.fit(X, dat)
    return mod.coef_.copy(), mod.predict(X), mod



def ltm2(tup, taus, method='lassoCV', **kwargs):
    wl, t, dat = tup.wl, tup.t, tup.data
    X = _make_base(tup, taus)
    kwargs['max_iter'] = 4000
    coefs = np.empty((taus.size, dat.shape[1]))
    m = lm.LassoCV(**kwargs)
    m.verbose = 1
    m.alphas = 10**np.arange(-5, -1, 1)
    m.n_jobs = 1
    print m.alphas
    for i in range(dat.shape[1]):
        m.fit(X, dat[:, i])
        coefs[:, i] = m.coef_.copy()
    return coefs, m

taus = np.logspace(log10(.05), log10(100), 100)
i = ti(.5)
#out = ltm2(dv.tup(wl, t[i:], df[i:, :]),
#          taus, 'elastic',
#          alpha=0.0001, l1_ratio=.9, normalize=1)

out = ltm2(dv.tup(wl, t[i:], df[i:, :]),
          taus)
#figure(2)
clf()

#plot(t[i:], out[1][:, 40].T, t[i:], df[i:, 40])
xscale('log')
def annotate(taus, data):
    idx = np.unique(np.hstack((sig.argrelmax(data, 0, 5), sig.argrelmin(data, 0, 5))))
    points = taus[idx]
    plt.vlines(points, np.zeros_like(points), data[idx])
    for i in idx:
        plt.text(taus[i], data[i] +np.sign(data[i])*0.3,
                 '{0: .2f}'.format(taus[i]), ha='center')

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm
def nice_map():
    figure(1)
    clf()
    ax = subplot(111)
    norm = SymLogNorm(linthresh=1)
    pcolormesh(wl, taus[dv.fi(taus, .1):], out[0][dv.fi(taus, .1):, :], cmap=cm.RdBu_r, norm=norm)
    yscale('log')
    plt.autoscale(1, 'both', 'tight')
    ax.set_ylim(.1,)
    ax.set_title('lifetimemap', fontsize=14)
    ax.set_ylabel('decay constant / ps')
    ax.set_xlabel('frequency / cm$^{-1}$')
    divider = make_axes_locatable(ax)

    axt = divider.append_axes("right", size=1.3, sharey=ax,
                              pad=0.15)
    axt.plot(out[0].sum(1), taus)
    plt.xscale('symlog')
    axt.set_xlabel('sum amp.')
    axt.set_ylim(.1, taus.max())
    axt.axvline(0, c='k', zorder=1.9)
    axt.axvline(-1, c='k', zorder=1.9)
    axt.axvline(1, c='k', zorder=1.9)
    axt.xaxis.set_major_locator(MaxNLocator(5))
    axt.yaxis.tick_right()
    savefig('2d-lfg-map.png')
    #axt.xaxis.tick_top()
    #plt.autoscale(1, 'y', 'tight')

#    axt = divider.append_axes("top", size=4, sharex=ax,
#                              pad=0.1)
#    axt.plot(wl, out[0].T[dv.fi(taus, 0.01):dv.fi(taus, 0.1), :].sum(0))
#    axt.plot(wl, out[0].T[dv.fi(taus, 0.15):dv.fi(taus, 0.3), :].sum(0))
#    axt.plot(wl, out[0].T[dv.fi(taus, 0.4):dv.fi(taus, 1), :].sum(0))
#    axt.plot(wl, 5*out[0].T[dv.fi(taus, 1):dv.fi(taus, 2), :].sum(0))
#    axt.plot(wl, 5*out[0].T[dv.fi(taus, 5):dv.fi(taus, 9), :].sum(0))
#    axt.plot(wl, 5*out[0].T[dv.fi(taus, 10):dv.fi(taus, 50), :].sum(0))
#    axt.xaxis.tick_top()
#    axt.axhline(0, c='k', zorder=1.9)
#    plt.autoscale(1, 'x', 'tight')
#    figure(3)
#    plot(wl, out[0].T[dv.fi(taus, 0.01):dv.fi(taus, 0.1), :].sum(0)+out[0].T[dv.fi(taus, 0.1):dv.fi(taus, 0.3), :].sum(0))
#    plot(wl, out[0].T[dv.fi(taus, 0.1):dv.fi(taus, 0.3), :].sum(0))
#    plot(wl, out[0].T[dv.fi(taus, 0.3):dv.fi(taus, 0.6), :].sum(0))
    #axt.set_yscale('log')

nice_map()
#plot(taus, s.coef_[14, :])
#annotate(taus, s.coef_[14, :])
#xscale('log')
def contiuenes_reg(X, ):
    pass

#f = np.loadtxt('ir abs.txt')
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

