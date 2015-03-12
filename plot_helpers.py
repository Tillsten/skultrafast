# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:35:22 2014

@author: tillsten
"""

import matplotlib.pyplot as plt
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
    inv_freq = True
    freq_unit = 'cm$^{-1}$'    

def vis_mode():
    global freq_label
    global inv_freq
    global freq_unit
    freq_label = 'Wavelength / nm'
    inv_freq = False
    freq_unit = 'nm'    

vis_mode()
time_label = 'Delay time  / ps'    
time_unit = 'ps'
sig_label = 'Absorbance change / mOD'
inv_freq = False
line_width = 1

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
        ax1.plot(t,u.T[i], label=str(i) )
        ax2.plot(wl,v[i] )
    ax1.legend()
    plt.subplot(313)
    plot_singular_values(d)
    plt.tight_layout()

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

def plot_trans(tup, wls, symlog=True, norm=False, marker=None, **kwargs):
    wl, t, d = tup.wl, tup.t, tup.data
    ulim = -np.inf
    llim = np.inf
    plotted_vals = []    
    for i in wls:
        idx = dv.fi(wl, i)
        dat = d[:, idx]
        if norm:
            dat = np.sign(dat[np.argmax(abs(dat))])* dat / abs(dat).max()
            
        plotted_vals.append(dat)
        plt.plot(t, dat, label='%.1f %s'%(wl[idx], freq_unit), marker=marker, **kwargs)
    
    ulim = np.percentile(plotted_vals, 99.) + 0.5
    llim = np.percentile(plotted_vals, 1.) - 0.5
    plt.xlabel(time_label)
    plt.ylabel(sig_label)
    plt.ylim(llim, ulim)
    if symlog:
        plt.xscale('symlog')
        plt.axvline(1, c='k', lw=0.5, zorder=1.9)
    plt.axhline(0, color='k', lw=0.5, zorder=1.9)
    plt.xlim(-.5,)
    plt.legend(loc='best', ncol=2, title='Wavelength')
    
def plot_ints(tup, wls, symlog=True, norm=False):
    wl, t, d = tup.wl, tup.t, tup.data
    ulim = -np.inf
    llim = np.inf
    plotted_vals = []    
    for i in wls:
        wl1, wl2 = i        
        idx1, idx2 = sorted([dv.fi(wl, wl1), dv.fi(wl, wl2)])        
        dat = d[:, idx1:idx2].mean(-1)
        if norm:
            dat = np.sign(dat[np.argmax(abs(dat))])* dat / abs(dat).max()
            
        
        plotted_vals.append(dat)
        plt.plot(t, dat, label='%.1f %s'%(wl[idx1:idx2].mean(), freq_unit), lw=line_width)
    
    ulim = np.percentile(plotted_vals, 99.) + 0.5
    llim = np.percentile(plotted_vals, 1.) - 0.5
    plt.xlabel(time_label)
    plt.ylabel(sig_label)
    plt.ylim(llim, ulim)
    if symlog:
        plt.xscale('symlog')
        plt.axvline(1, c='k', lw=0.5, zorder=1.9)
    plt.axhline(0, color='k', lw=0.5, zorder=1.9)
    plt.xlim(-.5,)
    plt.legend(loc='best', ncol=2, title='Wavelength')
    
def plot_diff(tup, t0, t_list):
    diff = tup.data - tup.data[dv.fi(tup.t, t0), :]
    plot_spec(dv.tup(tup.wl, tup.t, diff), t_list)
    
def plot_spec(tup, t_list):
    wl, t, d = tup.wl, tup.t, tup.data        
    for i in t_list:
        idx = dv.fi(t, i)
        dat = d[idx, :]        
        plt.plot(wl, dat, label='%.1f %s'%(t[idx], time_unit), lw=line_width)
    
    #ulim = np.percentile(plotted_vals, 98.) + 0.1
    #llim = np.percentile(plotted_vals, 2.) - 0.1
    plt.xlabel(freq_label)
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
    if norm is None:
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
    #norm = SymLogNorm(linthresh=0.3)
    norm = MidPointNorm(0)
    
    m = np.abs(coefs[:, :]).max()
    c = ax.pcolormesh(tup.wl, taus[:], coefs[:, :], cmap='bwr', vmin=-m, vmax=m, norm=norm)
    cb = plt.colorbar(c, pad=0.01)
    
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
    

def plot_freqs(tup, wl, from_t, to_t):    
    ti = dv.make_fi(tup.t)
    wi = dv.make_fi(tup.wl)
    tl = tup.t[ti(from_t):ti(to_t)]
    trans = tup.data[ti(from_t):ti(to_t), wi(wl)]
    ax1 = plt.subplot(311)
    ax1.plot(tl, trans)    
    dt = dv.polydetrend(trans, deg=2)
    ax1.plot(tl, -dt+trans)
    ax2 = plt.subplot(312)
    ax2.plot(tl, dt)
    ax3 = plt.subplot(313)
    f = abs(np.fft.fft(dt, 2*dt.size))
    freqs = np.fft.fftfreq(2*dt.size, tup.t[ti(from_t)-1]-tup.t[ti(from_t)])
    n = freqs.size/2+1
    ax3.plot(dv.fs2cm(1000/freqs[n:]), f[n:])
    ax3.set_xlabel('freq / cm$^{-1}$')


def plot_coef_spec(taus, wl, coefs, div):
    tau_coefs = coefs[:, :len(taus)]    
    div.append(taus.max()+1)
    ti = dv.make_fi(taus)
    last_idx = 0    
    non_zeros = ~(coefs.sum(0) == 0)
    for i in div:
        idx = ti(i) 
        cur_taus = taus[last_idx:idx]
        cur_nonzeros = non_zeros[last_idx:idx]
        lbl = "%.1f - %.1f ps"%(taus[last_idx], taus[idx])
        plt.plot(wl, tau_coefs[:, last_idx:idx].sum(-1), label=lbl)        
        last_idx = ti(i)        
    
    plt.plot(wl, coefs[:, -1])           
    plt.legend(title='Decay regions')
    lbl_spec()
    plt.title("Spectrum of lft-parts")

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
                

def fit_semiconductor(t, data, sav_n=11, sav_deg=4):    
    from scipy.signal import savgol_filter
    from scipy.optimize import leastsq
    ger =   data.sum(2).mean(-1).squeeze()    
    plt.subplot(121)
    plt.title('Germanium sum')
    plt.plot(t, ger[:,  0])
    plt.plot(t, ger[:,  1])
    plt.plot(t, savgol_filter(ger[:, 0], sav_n, sav_deg, 0))
    plt.plot(t, savgol_filter(ger[:, 1], sav_n, sav_deg, 0))
    plt.xlim(-1, 3)
    plt.subplot(122)
    plt.title('First dervitate')
    derv0 = savgol_filter(ger[:, 0, ], sav_n, sav_deg, 1)
    derv1 = savgol_filter(ger[:, 1, ], sav_n, sav_deg, 1)
    plt.plot(t , derv0)
    plt.plot(t , derv1)
    plt.xlim(-.8, .8)
    plt.ylim(0, 700)
    plt.minorticks_on()
    plt.grid(1)

    def gaussian(p, ch, res=True):
        
        i, j = dv.fi(t, -.8), dv.fi(t, .8)
        w = p[0]
        A = p[1]
        x0 = p[2]
        fit = A*np.exp(-(t[i:j]-x0)**2/(2*w**2))
        if res:
            return fit-savgol_filter(ger[:, ch, ], sav_n, sav_deg, 1)[i:j]
        else:
            return fit



    x0 = leastsq(gaussian, [.2, max(derv0), 0], 0)
    plt.plot(t[dv.fi(t, -.8):dv.fi(t, .8)], gaussian(x0[0], 0, 0), '--k', )
    plt.text(0.05, 0.8, 'x$_0$ = %.2f\nFWHM = %.2f\nA = %.1f\n'%(x0[0][2],2.35*x0[0][0], x0[0][1]),
             transform=plt.gca().transAxes)
    
    x0 = leastsq(gaussian, [.2, max(derv1), 0], 1)
    plt.plot(t[dv.fi(t, -.8):dv.fi(t, .8)], gaussian(x0[0], 1, 0), '--b', )
    
    plt.xlim(-.8, .8)
    plt.minorticks_on()
    plt.grid(0)
    plt.tight_layout()
    plt.text(0.5, 0.8, 'x$_0$ = %.2f\nFWHM = %.2f\nA = %.1f\n'%(x0[0][2],2.35*x0[0][0], x0[0][1]),
             transform=plt.gca().transAxes)