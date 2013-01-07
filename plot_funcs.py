# -*- coding: utf-8 *-*
units = {'x': ' nm', 'y': ' ps', 'z': '$\\Delta$OD'}
title = ""
import matplotlib.pyplot as plt
import numpy as np
import dv, data_io, zero_finding

plt.rcParams['font.size']=9
plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['legend.borderpad'] = 0.1
plt.rcParams['legend.columnspacing'] = 0.3
plt.rcParams['legend.labelspacing'] = 0.3
plt.rcParams['legend.loc'] = 'best'

def plot_das(fitter, plot_fastest=False, plot_coh=False ,normed=False):
    """Plots the decay-asscoiated  """        
    t_slice = fitter.model_disp+2
    fitter.last_para[t_slice:] = np.sort(fitter.last_para[t_slice:])
    fitter.res(fitter.last_para)
    
    if plot_coh and fitter.model_coh:
        ulim = fitter.num_exponentials
    else:
        ulim =- 4

    if plot_fastest:
        llim = 0 
    else:
        llim = 1 
    
    dat_to_plot= fitter.c.T[:,llim:ulim]
    if normed: 
        dat_to_plot = dat_to_plot/np.abs(dat_to_plot).max(0)
    plt.plot(fitter.wl, dat_to_plot, lw=2)
    plt.autoscale(1, tight=1)
    plt.axhline(0, color='grey', zorder=-10, ls='--')
    leg = np.round(fitter.last_para[2 + llim + fitter.model_disp:], 2)
    plt.legend([str(i)+ units['y'] for i in leg], labelspacing=0.25)
    plt.xlabel(units['x'])
    plt.ylabel(units['z'])
    if title:
        plt.title(title)

def plot_diagnostic(fitter):
    residuals=fitter.data-fitter.m.T
    u,s,v=np.linalg.svd(residuals)
    normed_res=residuals / np.std(residuals, 0)
    plt.subplot2grid((3, 3), (0, 0), 2, 3).imshow(normed_res, vmin=-3,
                                             vmax=3, aspect='auto')
    plt.subplot2grid((3, 3), (2, 0)).plot(fitter.t, u[:,:2])
    plt.subplot2grid((3, 3), (2, 1)).plot(fitter.wl, v.T[:,:2])
    ax=plt.subplot2grid((3, 3), (2, 2))
    ax.stem(range(1, 11), s[:10])
    ax.set_xlim(0, 12)

def plot_spectra(fitter, tp=None, num_spec=8, use_m=False):
    """
    Plots the transient spectra of an fitter object.
    """
    t = fitter.t
    tmin, tmax = t.min(),t.max()
    if tp is None: 
        tp = np.logspace(np.log10(0.100), np.log10(tmax), num=num_spec)
        tp = np.hstack([-0.5, -.1, 0, tp])
    tp = np.round(tp, 2)    
    t0 = fitter.last_para[fitter.model_disp]
    if use_m:
        data_used = fitter.m.T
    else:
        data_used = fitter.data
    specs = zero_finding.interpol(dv.tup(fitter.wl, fitter.t, data_used),
                             np.zeros(fitter.data.shape[1]), t0, tp).data
    
    plt.plot(fitter.wl,specs.T)    
    plt.legend([unicode(i)+u' '+units['y'] for i in np.round(tp,2)],
                ncol=2,  labelspacing=0.25)
    plt.axhline(0, color='grey', zorder=-10, ls='--')
    plt.autoscale(1, tight=1)
    plt.xlabel(units['x'])
    plt.ylabel(units['z'])
    if title:
        plt.title(title)


def plot_transients(fitter, wls, plot_fit=True, scale='linear'):
    wls = np.array(wls)
    idx = np.argmin(np.abs(wls[:,None]-fitter.wl[None,:]),1)
    print idx
    plt.plot(fitter.t + fitter.last_para[fitter.model_disp],
             fitter.data[:, idx], '^')
    names = [str(i) + u' ' + units['x'] for i in np.round(fitter.wl[idx])]
    plt.legend(names)
    if plot_fit and hasattr(fitter,'m'):
        plt.plot(fitter.t + fitter.last_para[fitter.model_disp], 
                 fitter.m.T[:, idx], 'k')
    plt.autoscale(1, tight=1)
    plt.xlabel(units['y'])
    plt.ylabel(units['z'])
    if scale != 'linear':
        plt.xscale(scale)
    if title:
        plt.title(title)

def plot_residuals(fitter, wls, scale='linear'):
    wls = np.array(wls)
    idx = np.argmin(np.abs(wls[:, None] - fitter.wl[None, :]), 1)
    plt.plot(fitter.t, (fitter.data - fitter.m.T)[:, idx], '-^')
    plt.legend([unicode(i) + u' ' + units['x'] for i in np.round(fitter.wl[idx])],
                 labelspacing=0.25)
    plt.autoscale(1, tight=1)
    plt.xlabel(units['y'])
    plt.ylabel(units['z'])
    if scale != 'linear':
        plt.xscale(scale)
    if title:
        plt.title(title)
        
def a4_overview(fitter, fname, plot_fastest=1):
    f=plt.figure(1, figsize=(8.3, 11.7))
    plt.subplot(321)
    plt.pcolormesh(fitter.wl, fitter.t, fitter.data)
    plt.autoscale(1, tight=1)
    plt.subplot(322)
    plt.imshow(fitter.residuals, aspect='auto')
    plt.autoscale(1, tight=1)
    plt.subplot(323)
    plot_das(fitter, plot_fastest)
    plt.subplot(324)
    plot_das(fitter, 1, normed=True)
    plt.subplot(325)
    plot_spectra(fitter)
    plt.subplot(326)    
    wl = fitter.wl
    ind = [int(round(i)) for i in np.linspace(wl.min(), wl.max(), 10)]    
    plot_transients(fitter, ind, scale='symlog')
    #plt.gcf().set_size_inches((8.2, 11.6))
    plt.subplots_adjust()
    plt.show()
    f.savefig(fname, dpi=600)    

def _plot_zero_finding(tup, raw_tn, fit_tn, cor):
    ax1 = plt.subplot(121)
    ax1.plot(tup.wl, raw_tn)    
    ax1.plot(tup.wl, fit_tn)    
    ax1.pcolormesh(tup.wl, tup.t, tup.data)
    ax1.set_ylim(fit_tn.min(), fit_tn.max())
    ax2 = plt.subplot(122)
    ax2.pcolormesh(cor.wl, cor.t, cor.data)
    ax2.set_ylim(fit_tn.min(), fit_tn.max())
    
    
def make_legend(p, err, n):
    dig = np.floor(np.log10(err))
    l = []
    for i in range(2, n + 2):
        val = str(np.around(p[i], -int(dig[i])))
        erri = str(np.ceil(round(err[i] * 10**(-dig[i]),3)) * 10**(dig[i]))
        s = ''.join(['$\\tau_', str(int(i - 1)), '=', val, '\\pm ', erri, ' $ps'])
        l.append(s)
    return l

def make_legend_noerr(p, err, n):
    dig = np.floor(np.log10(err))
    l = []
    for i in range(2,n+2):
        val = str(np.around(p[i], -int(dig[i])))        
        l.append('$\\tau_' + str(int(i - 1)) + '$=' + val + ' ps')
    return l