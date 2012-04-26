# -*- coding: utf-8 *-*
unitdict={'x': ' nm', 'y': ' ps', 'z': '$\\Delta$oD'}
import matplotlib.pyplot as plt
import numpy as np

def plot_das(fitter, plot_fastest=False, plot_coh=False ,normed=False):
    fitter.last_para[2:]=np.sort(fitter.last_para[2:])
    fitter.res(fitter.last_para)
    if plot_coh and fitter.model_coh:
        ulim=fitter.num_exponentials
    else:
        ulim=-4

    if plot_fastest:
        llim=0
    else:
        llim=1
    print llim, ulim
    dat_to_plot= fitter.c.T[:,llim:ulim]
    if normed: dat_to_plot=dat_to_plot/np.abs(dat_to_plot).max(0)
    plt.plot(fitter.wl,dat_to_plot,lw=2)
    plt.autoscale(1,tight=1)
    plt.legend(np.round(fitter.last_para[2+llim:],2))
    plt.xlabel(unitdict['x'])
    plt.ylabel(unitdict['z'])

def plot_diagnostic(fitter):
    residuals=fitter.data-fitter.m.T
    u,s,v=np.linalg.svd(residuals)
    plt.subplot2grid((3,3),(0,0),2,3).imshow(residuals,aspect='auto')
    plt.subplot2grid((3,3),(2,0)).plot(u[:,0])
    plt.subplot2grid((3,3),(2,1)).plot(v.T[:,0])
    ax=plt.subplot2grid((3,3),(2,2))
    ax.stem(range(1,11),s[:10])
    ax.set_xlim(0,12)

def plot_spectra(fitter,tp=None,num_spec=8):
    t=fitter.t
    tmin,tmax=t.min(),t.max()
    if tp==None: tp=np.logspace(np.log10(0.100),np.log10(tmax),num=num_spec)
    tp=np.round(tp,2)
    specs=dv.interpol(fitter.data,t,np.zeros(fitter.data.shape[1]),0,tp)
    plt.plot(fitter.wl,specs.T)
    plt.legend(tp,ncol=2)
    plt.autoscale(1,tight=1)
    plt.xlabel(unitdict['x'])
    plt.ylabel(unitdict['z'])


def plot_transients(fitter,wls, plot_fit=True,scale='linear'):
    wls=np.array(wls)
    idx=np.argmin(np.abs(wls[:,None]-fitter.wl[None,:]),1)
    plt.plot(fitter.t, fitter.data[:,idx],'^')
    plt.legend([unicode(i)+u' '+unitdict['x'] for i in np.round(fitter.wl[idx])])
    if plot_fit and hasattr(fitter,'m'):
        plt.plot(fitter.t,fitter.m.T[:,idx],'k')
    plt.autoscale(1,tight=1)
    plt.xlabel(unitdict['y'])
    plt.ylabel(unitdict['z'])
    if scale!='linear':
        plt.xscale(scale)
