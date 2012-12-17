# -*- coding: utf-8 *-*
unitdict={'x': ' nm', 'y': ' ps', 'z': '$\\Delta$OD'}
title=""
import matplotlib.pyplot as plt
import numpy as np
import dv

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
    if normed: dat_to_plot=dat_to_plot/np.abs(dat_to_plot).max(0)
    plt.plot(fitter.wl,dat_to_plot,lw=2)
    plt.autoscale(1,tight=1)
    plt.axhline(0,color='grey',zorder=-10,ls='--')
    leg=np.round(fitter.last_para[2+llim+fitter.model_disp:],2)
    plt.legend([str(i)+ unitdict['y'] for i in leg])
    plt.xlabel(unitdict['x'])
    plt.ylabel(unitdict['z'])
    if title:
        plt.title(title)

def plot_diagnostic(fitter):
    residuals=fitter.data-fitter.m.T
    u,s,v=np.linalg.svd(residuals)
    normed_res=residuals/np.std(residuals,0)
    plt.subplot2grid((3,3),(0,0),2,3).imshow(normed_res,vmin=-5,
                                             vmax=5,aspect='auto')
    plt.subplot2grid((3,3),(2,0)).plot(fitter.t, u[:,:2])
    plt.subplot2grid((3,3),(2,1)).plot(fitter.wl,v.T[:,:2])
    ax=plt.subplot2grid((3,3),(2,2))
    ax.stem(range(1,11),s[:10])
    ax.set_xlim(0,12)

def plot_spectra(fitter,tp=None,num_spec=8,loc=4,use_m=False):
    t=fitter.t
    tmin,tmax=t.min(),t.max()
    if tp==None: tp=np.logspace(np.log10(0.100),np.log10(tmax),num=num_spec)
    tp=np.round(tp,2)    
    t0=fitter.last_para[fitter.model_disp]
    if use_m:
        data_used=fitter.m.T
    else:
        data_used=fitter.data
    specs=dv.interpol(data_used,fitter.t,np.zeros(fitter.data.shape[1]),t0,tp)
    plt.plot(fitter.wl,specs.T)    
    plt.legend([unicode(i)+u' '+unitdict['y'] for i in np.round(tp,2)],ncol=2,loc=loc)
    plt.axhline(0,color='grey',zorder=-10,ls='--')
    plt.autoscale(1,tight=1)
    plt.xlabel(unitdict['x'])
    plt.ylabel(unitdict['z'])
    if title:
        plt.title(title)


def plot_transients(fitter, wls, plot_fit=True, scale='linear'):
    wls = np.array(wls)
    idx = np.argmin(np.abs(wls[:,None]-fitter.wl[None,:]),1)
    plt.plot(fitter.t + fitter.last_para[fitter.model_disp],
             fitter.data[:, idx], '^')
    plt.legend([unicode(i) + u' ' + unitdict['x'] for i in np.round(fitter.wl[idx])])
    if plot_fit and hasattr(fitter,'m'):
        plt.plot(fitter.t + fitter.last_para[fitter.model_disp], 
                 fitter.m.T[:, idx], 'k')
    plt.autoscale(1, tight=1)
    plt.xlabel(unitdict['y'])
    plt.ylabel(unitdict['z'])
    if scale != 'linear':
        plt.xscale(scale)
    if title:
        plt.title(title)

def plot_residuals(fitter, wls, scale='linear'):
    wls = np.array(wls)
    idx = np.argmin(np.abs(wls[:, None] - fitter.wl[None, :]), 1)
    plt.plot(fitter.t, (fitter.data - fitter.m.T)[:, idx], '^')
    plt.legend([unicode(i) + u' ' + unitdict['x'] for i in np.round(fitter.wl[idx])])
    plt.autoscale(1, tight=1)
    plt.xlabel(unitdict['y'])
    plt.ylabel(unitdict['z'])
    if scale != 'linear':
        plt.xscale(scale)
    if title:
        plt.title(title)
        
        
        

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