# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:35:22 2014

@author: tillsten
"""

import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 120
plt.rcParams['figure.figsize'] = (6, 4)
import skultrafast.dv as dv
import numpy as np

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

tableau20 = [(r/255., g/255., b/255.) for r,g,b, in tableau20]
plt.rcParams['axes.color_cycle'] =  tableau20[::2]
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['savefig.dpi'] = 100

from matplotlib.colors import Normalize, SymLogNorm
import  matplotlib.cbook as cbook
ma = np.ma



class Data(object):
    def __init__(self, tup, name=''):
        self.tup = tup
        self.name = name

    def plot_spec(self, t_list, *args, **kwargs):
        plot_spec(self.tup, t_list, *args, **kwargs)

    def plot_trans(self, wl_list, *args, **kwargs):
        plot_trans(self.tup, wl_list, *args, **kwargs)
        
    def plot_map(self, *args, **kwargs):
        tup = self.tup
        nice_map(tup.wl, tup.t, tup.data, *args, **kwargs)
        
        
def ir_mode():
    global freq_label
    global inv_freq
    global freq_unit
    freq_label = 'Wavenumber / cm$^{-1}$'
    inv_freq = False
    freq_unit = 'cm$^{-1}$'    


def vis_mode():
    global freq_label
    global inv_freq
    global freq_unit
    freq_label = 'Wavelengths / nm'
    inv_freq = False
    freq_unit = 'nm'    




vis_mode()
time_label = 'Delay time  / ps'    
time_unit = 'ps'
sig_label = 'Absorbance change / mOD'
inv_freq = False

def plot_singular_values(dat):
    u, s, v = np.linalg.svd(dat)
    plt.vlines(np.arange(len(s)), 0, s, lw=3)
    plt.plot(np.arange(len(s)), s, 'o')
    plt.xlim(-1, 30)
    plt.ylim(1, )
    plt.yscale('log')
    plt.minorticks_on()
    plt.title('Singular values')
    plt.xlabel('N')
    plt.ylabel('Value')

def plot_svd_components(tup, n=4, from_t = None):    
    wl, t, d = tup.wl, tup.t, tup.data
    if from_t:
        idx = dv.fi(t, from_t)
        t = t[idx:]
        d = d[idx:, :]
    u, s, v = np.linalg.svd(d)
    ax1 = plt.subplot(311)
    ax1.set_xlim(-1, t.max())
    ax1.set_xscale('symlog')    
    lbl_trans()
    plt.minorticks_off()
    ax2 = plt.subplot(312)
    lbl_spec()
    plt.ylabel('')
    for i in range(n):
        ax1.plot(t,u.T[i] )
        ax2.plot(wl,v[i] )
    plt.subplot(313)
    plot_singular_values(d)

def make_angle_plot(wl, t, para, senk, t_range):
    p = para
    s = senk
    t0, t1 = dv.fi(t, t_range[0]), dv.fi(t, t_range[1])
    pd = p[t0:t1, :].mean(0)
    sd = s[t0:t1, :].mean(0)

    ax = plt.subplot(211)
    ax.plot(wl, pd)
    ax.plot(wl, sd)

    ax.axhline(0, c='k')
    ax.legend(['Parallel', 'Perpendicular'], columnspacing=0.3, ncol=2, frameon=0)

    ax.xaxis.tick_top()
    ax.set_ylabel(sig_label)
    ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.1,  'Signal average\nfor %.1f...%.0f ps'%t_range,
            transform=ax.transAxes)
            #horizontalalignment='center')

    ax2 = plt.subplot(212, sharex=ax)
    d = pd/sd
    ang = np.arccos(np.sqrt((2*d-1)/(d+2)))/np.pi*180

    ax2.plot(wl, ang, 'o-')
    ax2.set_ylim(0, 90)
    ax2.set_ylabel('Angle / Degrees')
    ax3 = plt.twinx()
    ax3.plot(wl, ang, lw=0)
    ax2.invert_xaxis()
    f = lambda x: "%.1f"%(to_dichro(float(x)/180.*np.pi))
    ax2.set_ylim(0, 90)
    def to_angle(d):
        return np.arccos(np.sqrt((2*d-1)/(d+2)))/np.pi*180
    def to_dichro(x):
        return (1+2*np.cos(x)**2)/(2-np.cos(x)**2)
    n_ticks = ax2.yaxis.get_ticklocs()
    ratio_ticks = np.array([0.5, 0.7, 1., 1.5, 2., 2.5, 3.])
    ax3.yaxis.set_ticks(to_angle(ratio_ticks))
    ax3.yaxis.set_ticklabels([i for i in ratio_ticks])
    ax3.set_ylabel('$A_\\parallel  / A_\\perp$')
    ax2.set_xlabel()
    ax2.set_title('Angle calculated from dichroic ratio', fontsize='x-small')
    plt.tight_layout(rect=[0, 0, 1, 1], h_pad=0)
    return ax, ax2, ax3

def make_angle_plot2(wl, t, para, senk, t_range):
    p = para
    s = senk
    t0, t1 = dv.fi(t, t_range[0]), dv.fi(t, t_range[1])
    pd = p[t0:t1, :].mean(0)
    sd = s[t0:t1, :].mean(0)

    ax = plt.subplot(111)
    ax.plot(wl, pd)
    ax.plot(wl, sd)
    ax.plot([], [], 's-', color='k')
    ax.axhline(0, c='k', zorder=1.9)

    ax.invert_xaxis()
    #ax.xaxis.tick_top()
    ax.set_ylabel(sig_label)
    ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.05,  'Signal average\nfor %.1f...%.1f ps'%t_range,
            transform=ax.transAxes)
    ax.legend(['parallel', 'perpendicular', 'angle'], columnspacing=0.3, ncol=3, frameon=0)
            #horizontalalignment='center')

    ax2 = plt.twinx(ax)
    d = pd/sd
    ang = np.arccos(np.sqrt((2*d-1)/(d+2)))/np.pi*180

    ax2.plot(wl, ang, 's-', color='k')
    for i in np.arange(10, 90, 10):
        ax2.axhline(i, c='gray', linestyle='-.', zorder=1.8, lw=.5, alpha=0.5)
    ax2.set_ylim(0, 90)
    ax2.set_ylabel('angle / degrees')

def lbl_spec(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_xlabel(freq_label)
    ax.set_ylabel(sig_label)
    if inv_freq:
        ax.invert_xaxis()
    ax.axhline(0, c='k', zorder=1.5)

    plt.minorticks_on()


def lbl_trans(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_xlabel(time_label)
    ax.set_ylabel(sig_label)
    ax.axhline(0, c='k', zorder=1.5)    
    plt.minorticks_on()

def plot_trans(tup, wls, symlog=True):
    wl, t, d = tup.wl, tup.t, tup.data
    ulim = -np.inf
    llim = np.inf
    plotted_vals = []
    for i in wls:
        idx = dv.fi(wl, i)
        dat = d[:, idx]
        plotted_vals.append(dat)
        plt.plot(t, dat, label='%.1f %s'%(wl[idx], freq_unit), lw=2)
    
    ulim = np.percentile(plotted_vals, 98.) + 0.1
    llim = np.percentile(plotted_vals, 2.) - 0.1
    plt.xlabel(time_label)
    plt.ylabel(sig_label)
    plt.ylim(llim, ulim)
    if symlog:
        plt.xscale('symlog')
    plt.axhline(0, color='k', lw=0.5, zorder=1.9)
    plt.xlim(-.5,)
    plt.legend(loc='best', ncol=2)
    
    
def plot_spec(tup, t_list):
    wl, t, d = tup.wl, tup.t, tup.data        
    for i in t_list:
        idx = dv.fi(t, i)
        dat = d[idx, :]        
        plt.plot(wl, dat, label='%.1f %s'%(t[idx], time_unit), lw=2)
    
    #ulim = np.percentile(plotted_vals, 98.) + 0.1
    #llim = np.percentile(plotted_vals, 2.) - 0.1
    plt.xlabel(time_label)
    plt.ylabel(sig_label)
    plt.autoscale(1, 'x', 1)        
    plt.axhline(0, color='k', lw=0.5, zorder=1.9)    
    plt.legend(loc='best', ncol=2,  title='Delay time')
    
    
def mean_spec(wl, t, p, t_range, ax=None, color=plt.rcParams['axes.color_cycle'],
              markers = ['o', '^']):
    if ax is None:
       ax = plt.gca()
    if not isinstance(p, list):
        p = [p]
    if not isinstance(t_range, list):
        t_range = [t_range]
    for j, (x, y) in enumerate(t_range):
        for i, d in enumerate(p):
            t0, t1 = dv.fi(t, x), dv.fi(t, y)
            pd = d[t0:t1, :].mean(0)
            ax.plot(wl, pd, color=color[j], marker=markers[i],
                    mec='none', ms=5)

        ax.text(0.8, 0.1+j*0.04,'%.1f ps'%t[t0:t1].mean(),
                color=color[j],
                transform=ax.transAxes)

    lbl_spec(ax)

    if len(p) == 1:
        ax.set_title('mean signal from {0:.1f} to {1:.1f} ps'.format(t[t0], t[t1]))

def nice_map(wl, t, d, lvls=20, linthresh=10, linscale=1, norm=None, 
             **kwargs):
    if not norm:
        norm = SymLogNorm(linthresh, linscale=linscale)
    con = plt.contourf(wl, t, d, lvls, norm=norm, cmap='coolwarm', **kwargs)
    cb = plt.colorbar(pad=0.02)
    cb.set_label(sig_label)
    plt.contour(wl, t, d, lvls, norm=norm, colors='black', lw=.5, linestyles='solid')

    plt.yscale('symlog', linthreshy=1, linscaley=1, suby=[2,3,4,5,6,7,8,9])
    plt.ylim(-.5, )
    plt.xlabel(freq_label)
    plt.ylabel(time_label)
    return con 


def nice_lft_map(tup, taus, coefs):
    plt.figure(1, figsize=(6, 4))    
    ax = plt.subplot(111)
    norm = SymLogNorm(linthresh=.1)
    #norm = MidPointNorm(0)
    
    m = np.abs(coefs[:, :]).max()
    ax.pcolormesh(tup.wl, taus[:], coefs[:, :], cmap=cm.coolwarm, vmin=-m, vmax=m, norm=norm)
    cb = ax.colorbar(pad=0.01)
    
    cb.set_label('Amplitude')
    ax.set_yscale('log')
    plt.autoscale(1, 'both', 'tight')
    ax.set_ylim(None, 60)
    ax.set_xlabel(freq_label)
    ax.set_ylabel('Decay constant / ps')
    ax.set_title('Lifetime-map')
    if inv_freq:
        ax.invert_xaxis()
    divider = make_axes_locatable(ax)
    if 0:
        axt = divider.append_axes("right", size=1.3, sharey=ax, 
                                  pad=0.1)

        pos = np.where(out[0]>0, out[0], 0).sum(0)
        neg = np.where(out[0]<0, out[0], 0).sum(0)
        print pos.shape
        axt.plot(pos, taus)
        axt.plot(neg, taus)
        #axt.plot(out[0].T[:, wi(1513):].sum(1), taus)
        #axt.plot(3*out[0].T[:, :wi(1513)].sum(1), taus)
        plt.autoscale(1, 'both', 'tight')
    if 0:
        axt = divider.append_axes("top", size=1, sharex=ax, 
                              pad=0.1)
        axt.plot(tup.wl, out[0].T[:dv.fi(taus, 0.2), :].sum(0))
        axt.plot(tup.wl, out[0].T[dv.fi(taus, 0.3):dv.fi(taus, 1), :].sum(0))
        axt.plot(tup.wl, out[0].T[dv.fi(taus, 1):dv.fi(taus, 5), :].sum(0))
        axt.plot(tup.wl, out[0].T[dv.fi(taus, 5):dv.fi(taus, 10), :].sum(0))

        axt.xaxis.tick_top()
        axt.axhline(0, c='k', zorder=1.9)
    plt.autoscale(1, 'x', 'tight')
    
    


class MidPointNorm(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = np.ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint