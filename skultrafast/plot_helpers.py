# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:35:22 2014

@author: tillsten
"""
import string
import math
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import skultrafast.dv as dv
from skultrafast.unit_conversions import fs2cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, SymLogNorm
import matplotlib.cbook as cbook
from scipy import interpolate
import lmfit

ma = np.ma

linewidth = 2


def ir_mode():
    global freq_label
    global inv_freq
    global freq_unit
    freq_label = u'Wavenumber [cm$^{-1}$]'
    inv_freq = True
    freq_unit = u'cm$^{-1}$'


def vis_mode():
    global freq_label
    global inv_freq
    global freq_unit
    freq_label = 'Wavelength [nm]'
    inv_freq = False
    freq_unit = 'nm'


vis_mode()
time_label = 'Delay time [ps]'
time_unit = 'ps'
sig_label = 'Absorbance change [mOD]'
vib_label = 'Wavenumber  [cm$^{-1}$]'

freq_label = 'Wavelength [nm]'
inv_freq = False
freq_unit = 'nm'

line_width = 1


def plot_singular_values(dat):
    u, s, v = np.linalg.svd(dat)
    plt.vlines(np.arange(len(s)), 0, s, lw=3)
    plt.plot(np.arange(len(s)), s, 'o')

    plt.xlim(-1, 30)
    #plt.ylim(-1, )
    plt.yscale('log')
    plt.minorticks_on()
    plt.title('Singular values')
    plt.xlabel('N')
    plt.ylabel('Value')


def make_dual_axis(ax: plt.Axes = None, axis='x', unit='nm', minor_ticks=True):
    if ax is None:
        ax = plt.gca()

    if axis == 'x':
        pseudo_ax = ax.twiny()
        limits = ax.get_xlim()
        u, l = 1e7 / np.array(limits)
        pseudo_ax.set_xlim(limits)
        sub_axis = pseudo_ax.xaxis

    elif axis == 'y':
        pseudo_ax = ax.twinx()
        limits = ax.get_ylim()
        u, l = 1e7 / np.array(limits)
        pseudo_ax.set_ylim(limits)
        sub_axis = pseudo_ax.yaxis
    else:
        raise ValueError('axis must be either x or y.')

    def conv(x, y):
        return '%.0f' % (1e7/x)

    ff = plt.FuncFormatter(conv)
    sub_axis.set_major_formatter(ff)
    major = [1000, 500, 200, 100, 50]
    minor = [200, 100, 50, 25, 10]
    for x, m in zip(major, minor):
        a, b = math.ceil(u / x), math.ceil(l / x)
        n = abs(b - a)
        if n > 4:
            ticks = np.arange(
                a * x,
                b * x,
                x,
            )

            a, b = math.floor(u / m), math.floor(l / m)
            min_ticks = np.arange(a * m, b * m, m)

            break

    sub_axis.set_ticks(1e7 / ticks)
    sub_axis.set_ticks(1e7 / min_ticks, minor=True)
    if minor_ticks:
        ax.minorticks_on()
        # pseudo_ax.minorticks_on()
    if unit == 'nm':
        sub_axis.set_label('Wavelengths [nm]')
    elif unit == 'cm':
        sub_axis.set_label('Wavenumber [1/cm]')


def plot_svd_components(tup, n=4, from_t=None):
    wl, t, d = tup.wl, tup.t, tup.data
    if from_t:
        idx = dv.fi(t, from_t)
        t = t[idx:]
        d = d[idx:, :]
    u, s, v = np.linalg.svd(d)
    ax1: plt.Axes = plt.subplot(311)
    ax1.set_xlim(-1, t.max())

    lbl_trans()
    plt.minorticks_off()
    ax1.set_xscale('symlog')
    ax2 = plt.subplot(312)
    lbl_spec()
    plt.ylabel('')
    for i in range(n):
        ax1.plot(t, u.T[i], label=str(i))
        ax2.plot(wl, v[i])
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
    ax.text(0.05,
            0.1,
            'Signal average\nfor %.1f...%.0f ps' % t_range,
            transform=ax.transAxes)
    # horizontalalignment='center')

    ax2 = plt.subplot(212, sharex=ax)
    d = pd / sd
    ang = np.arccos(np.sqrt((2*d - 1) / (d+2))) / np.pi * 180

    ax2.plot(wl, ang, 'o-')
    ax2.set_ylim(0, 90)
    ax2.set_ylabel('Angle / Degrees')
    ax3 = plt.twinx()
    ax3.plot(wl, ang, lw=0)
    ax2.invert_xaxis()

    def f(x):
        return "%.1f" % (to_dichro(float(x) / 180. * np.pi))

    ax2.set_ylim(0, 90)

    def to_angle(d):
        return np.arccos(np.sqrt((2*d - 1) / (d+2))) / np.pi * 180

    def to_dichro(x):
        return (1 + 2 * np.cos(x)**2) / (2 - np.cos(x)**2)

    n_ticks = ax2.yaxis.get_ticklocs()
    ratio_ticks = np.array([0.5, 0.7, 1., 1.5, 2., 2.5, 3.])
    ax3.yaxis.set_ticks(to_angle(ratio_ticks))
    ax3.yaxis.set_ticklabels([i for i in ratio_ticks])
    ax3.set_ylabel('$A_\\parallel  / A_\\perp$')
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
    # ax.xaxis.tick_top()
    ax.set_ylabel(sig_label)
    ax.xaxis.set_label_position('top')
    ax.text(0.05,
            0.05,
            'Signal average\nfor %.1f...%.1f ps' % t_range,
            transform=ax.transAxes)
    ax.legend(['parallel', 'perpendicular', 'angle'],
              columnspacing=0.3,
              ncol=3,
              frameon=0)
    # horizontalalignment='center')

    ax2 = plt.twinx(ax)
    d = pd / sd
    ang = np.arccos(np.sqrt((2*d - 1) / (d+2))) / np.pi * 180

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
        x, y = ax.get_xlim()
        ax.set_xlim(sorted((x, y))[::-1])
    c = plt.rcParams['grid.color']
    ax.axhline(0, c=c, zorder=1.5)
    ax.minorticks_on()
    # plt.minorticks_on()


def lbl_trans(ax=None, use_symlog=True):
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel(time_label)
    ax.set_ylabel(sig_label)
    c = plt.rcParams['grid.color']
    ax.axhline(0, c=c, zorder=1.5)
    if use_symlog:
        symticks(ax, axis='x')
        ax.axvline(1, c='k', lw=0.5, zorder=1.5)
        ax.set_xlim(-.2)
    else:
        ax.minorticks_on()


def lbl_map(ax=None, use_symlog=True):
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel(freq_label)
    ax.set_ylabel(time_label)

    if use_symlog:
        symticks(ax, axis='y')
        ax.axhline(1, c='k', lw=0.5, zorder=1.5)
        ax.set_ylim(-.5)


def plot_trans(tup, wls, symlog=True, norm=False, marker=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    wl, t, d = tup.wl, tup.t, tup.data
    ulim = -np.inf
    llim = np.inf
    plotted_vals = []
    l = []
    for i in wls:
        idx = dv.fi(wl, i)
        dat = d[:, idx]
        if norm is True:
            dat = np.sign(dat[np.argmax(abs(dat))]) * dat / abs(dat).max()
        elif norm is False:
            pass
        else:
            dat = dat / dat[dv.fi(t, norm)]

        plotted_vals.append(dat)
        l.extend(
            ax.plot(t,
                    dat,
                    label='%.1f %s' % (wl[idx], freq_unit),
                    marker=marker,
                    **kwargs))

    ulim = np.percentile(plotted_vals, 99.) + 0.5
    llim = np.percentile(plotted_vals, 1.) - 0.5
    ax.set_xlabel(time_label)
    ax.set_ylabel(sig_label)
    #plt.ylim(llim, ulim)
    if symlog:
        ax.set_xscale('symlog', linthresh=1)
        ax.axvline(1, c='k', lw=0.5, zorder=1.9)
        symticks(ax)
    ax.axhline(0, color='k', lw=0.5, zorder=1.9)
    ax.set_xlim(-.5, )
    ax.legend(loc='best', ncol=2, title='Wavelength')
    return l


def mean_tup(tup, time):
    wl, t, d = tup.wl, tup.t, tup.data
    new_dat = tup.data / tup.data[dv.fi(t, time), :]
    return dv.tup(wl, t, new_dat)


def plot_ints(tup,
              wls,
              factors=None,
              symlog=True,
              norm=False,
              is_wavelength=True,
              ax=None,
              **kwargs):
    if ax is None:
        ax = plt.gca()
    wl, t, d = tup.wl, tup.t, tup.data
    lines = []
    plotted_vals = []
    for i in wls:
        dat = dv.spec_int(tup, i, is_wavelength)
        if norm is True:
            dat = np.sign(dat[np.argmax(abs(dat))]) * dat / abs(dat).max()
        elif norm is False:
            pass
        else:
            dat = dat / dat[dv.fi(t, norm)]

        plotted_vals.append(dat)
        idx1, idx2 = dv.fi(wl, i)
        label = 'From {0: .1f} - {1: .1f} {2}'.format(wl[idx1], wl[idx2], freq_unit)
        lines += ax.plot(t, dat, label=label, **kwargs)

    lbl_trans(ax)
    ax.set_xlim(-.5, )
    if symlog:
        ax.set_xscale('symlog')
        ax.axvline(1, c='k', lw=0.5, zorder=1.9)
        symticks(ax)
    ax.axhline(0, color='k', lw=0.5, zorder=1.9)

    ax.legend(loc='best', ncol=1)
    return lines


def plot_diff(tup, t0, t_list, **kwargs):
    diff = tup.data - tup.data[dv.fi(tup.t, t0), :]
    plot_spec(dv.tup(tup.wl, tup.t, diff), t_list, **kwargs)


def time_formatter(time, unit='ps'):
    mag = np.floor(np.log10(abs(time)))
    if time > 5:
        return '%.0f %s' % (time, unit)
    if time > 1:
        return '%.1f %s' % (time, unit)
    else:
        return '%1.2f %s' % (time, unit)


def plot_spec(tup, t_list, ax=None, norm=False, **kwargs):
    if ax is None:
        ax = plt.gca()
    wl, t, d = tup.wl, tup.t, tup.data
    li = []
    for i in t_list:
        idx = dv.fi(t, i)
        dat = d[idx, :]
        if norm:
            dat = dat / abs(dat).max()
        li += ax.plot(wl, dat, label=time_formatter(t[idx], time_unit), **kwargs)

    #ulim = np.percentile(plotted_vals, 98.) + 0.1
    #llim = np.percentile(plotted_vals, 2.) - 0.1
    ax.set_xlabel(freq_label)
    ax.set_ylabel(sig_label)
    ax.autoscale(1, 'x', 1)
    ax.axhline(0, color='k', lw=0.5, zorder=1.9)
    ax.legend(loc='best', ncol=2, title='Delay time')
    return li


def mean_spec(wl, t, p, t_range, ax=None, pos=(0.1, 0.1), markers=['o', '^']):
    if ax is None:
        ax = plt.gca()
    if not isinstance(p, list):
        p = [p]
    if not isinstance(t_range, list):
        t_range = [t_range]
    l = []
    for j, (x, y) in enumerate(t_range):
        for i, d in enumerate(p):
            t0, t1 = dv.fi(t, x), dv.fi(t, y)
            pd = np.mean(d[t0:t1, :], 0)
            lw = 2 if i == 0 else 1
            l += ax.plot(wl,
                         pd,
                         color='C%d' % j,
                         marker=markers[i],
                         lw=lw,
                         mec='none',
                         ms=3)

        ax.text(pos[0],
                pos[1] + j*0.07,
                '%.1f - %.1f ps' % (t[t0], t[t1]),
                color='C%d' % j,
                transform=ax.transAxes)

    lbl_spec(ax)

    if len(t_range) == 1:
        print(len(p))
        ax.set_title('mean signal from {0:.1f} to {1:.1f} ps'.format(t[t0], t[t1]))
    return l


def nice_map(wl,
             t,
             d,
             lvls=20,
             linthresh=10,
             linscale=1,
             norm=None,
             linscaley=1,
             cmap='coolwarm',
             **kwargs):
    if norm is None:
        norm = SymLogNorm(linthresh, linscale=linscale)
    con = plt.contourf(wl, t, d, lvls, norm=norm, cmap=cmap, **kwargs)
    cb = plt.colorbar(pad=0.02)
    cb.set_label(sig_label)
    plt.contour(wl, t, d, lvls, norm=norm, colors='black', lw=.5, linestyles='solid')

    plt.yscale('symlog', linthresh=1, linscale=linscaley, suby=[2, 3, 4, 5, 6, 7, 8, 9])
    plt.ylim(-.5, )
    plt.xlabel(freq_label)
    plt.ylabel(time_label)
    return con


def nice_lft_map(tup, taus, coefs, show_sums=False, **kwargs):
    cmap = kwargs.pop('cmap', 'seismic')
    plt.figure(1, figsize=(6, 4))
    ax = plt.subplot(111)
    #norm = SymLogNorm(linthresh=0.3)
    norm = kwargs.pop('norm', MidPointNorm(0))

    m = np.abs(coefs[:, :]).max()
    c = ax.pcolormesh(tup.wl,
                      taus[:],
                      coefs[:, :],
                      cmap=cmap,
                      vmin=-m,
                      vmax=m,
                      norm=norm,
                      **kwargs)
    cb = plt.colorbar(c, pad=0.01)

    cb.set_label('Amplitude')
    ax.set_yscale('log')
    plt.autoscale(1, 'both', 'tight')
    #ax.set_ylim(None, 60)

    plt.minorticks_on()
    ax.set_xlabel(freq_label)
    ax.set_ylabel('Decay constant [ps]')
    if inv_freq:
        ax.invert_xaxis()
    divider = make_axes_locatable(ax)
    if show_sums:
        axt = divider.append_axes("left", size=.5, sharey=ax, pad=0.05)
        pos = np.where(coefs > 0, coefs, 0).sum(1)
        neg = np.where(coefs < 0, coefs, 0).sum(1)
        axt.plot(pos[:len(taus)], taus, 'r', label='pos.')
        axt.plot(-neg[:len(taus)], taus, 'b', label='neg.')
        axt.plot(abs(coefs).sum(1)[:len(taus)], taus, 'k', label='abs.')
        axt.legend(frameon=False, loc='best')
        axt.invert_xaxis()
        #axt.plot(out[0].T[:, wi(1513):].sum(1), taus)
        #axt.plot(3*out[0].T[:, :wi(1513)].sum(1), taus)
        #plt.autoscale(1, 'y', 'tight')
        axt.set_ylabel('Decay constant [ps]')
        axt.xaxis.set_minor_locator(plt.NullLocator())
        axt.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(labelleft=0)
    else:
        ax.set_ylabel('Decay constant [ps]')
    if 0:
        axt = divider.append_axes("top", size=1, sharex=ax, pad=0.1)
        axt.plot(tup.wl, out[0].T[:dv.fi(taus, 0.2), :].sum(0))
        axt.plot(tup.wl, out[0].T[dv.fi(taus, 0.3):dv.fi(taus, 1), :].sum(0))
        axt.plot(tup.wl, out[0].T[dv.fi(taus, 1):dv.fi(taus, 5), :].sum(0))
        axt.plot(tup.wl, out[0].T[dv.fi(taus, 5):dv.fi(taus, 10), :].sum(0))

        axt.xaxis.tick_top()
        axt.axhline(0, c='k', zorder=1.9)
    plt.autoscale(1, 'both', 'tight')


def plot_freqs(tup, wl, from_t, to_t, taus=[1]):
    ti = dv.make_fi(tup.t)
    wi = dv.make_fi(tup.wl)
    tl = tup.t[ti(from_t):ti(to_t)]
    trans = tup.data[ti(from_t):ti(to_t), wi(wl)]

    #ax1 = plt.subplot(311)
    #ax1.plot(tl, trans)
    dt = dv.exp_detrend(trans, tl, taus)

    #ax1.plot(tl, -dt+trans)
    #ax2 = plt.subplot(312)
    ax3 = plt.subplot(111)

    f = abs(np.fft.fft(np.kaiser(2 * dt.size, 2) * dt, dt.size * 2))**2
    freqs = np.fft.fftfreq(dt.size * 2, tup.t[ti(from_t) + 1] - tup.t[ti(from_t)])
    n = freqs.size // 2
    ax3.plot(dv.fs2cm(1000 / freqs[1:n]), f[1:n])
    ax3.set_xlabel('freq / cm$^{-1}$')
    return dv.fs2cm(1000 / freqs[1:n]), f[1:n]


def plot_fft(x, y, min_amp=0.2, order=1, padding=2, power=1, ax=None):
    from scipy.signal import argrelmax
    if ax is None:
        ax = plt.gca()
    f = abs(np.fft.fft(y, padding * y.size))**power
    freqs = np.fft.fftfreq(padding * x.size, x[1] - x[0])
    n = freqs.size // 2 + 1
    fr_cm = -fs2cm(1000 / freqs[n:])

    ax.plot(fr_cm, f[n:])
    ax.set_xlabel('Wavenumber / cm$^{-1}$')
    ax.set_ylabel('FFT amplitude')
    for i in argrelmax(f[n:], order=1)[0]:
        if f[n + i] > min_amp:
            ax.text(fr_cm[i], f[n + i], '%d' % fr_cm[i], ha='center', va='bottom')


def plot_coef_spec(taus, wl, coefs, div):
    tau_coefs = coefs[:, :len(taus)]
    div.append(taus.max() + 1)
    ti = dv.make_fi(taus)
    last_idx = 0
    non_zeros = ~(coefs.sum(0) == 0)
    for i in div:
        idx = ti(i)
        cur_taus = taus[last_idx:idx]
        cur_nonzeros = non_zeros[last_idx:idx]
        lbl = "%.1f - %.1f ps" % (taus[last_idx], taus[idx])
        plt.plot(wl, tau_coefs[:, last_idx:idx].sum(-1), label=lbl)
        last_idx = ti(i)

    plt.plot(wl, coefs[:, -1])
    plt.legend(title='Decay regions', loc='best')
    lbl_spec()
    plt.title("Spectrum of lft-parts")


class MidPointNorm(Normalize):

    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self, vmin, vmax, clip)
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
            result.fill(0)  # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax), mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            # First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat > 0] /= abs(vmax - midpoint)
            resdat[resdat < 0] /= abs(vmin - midpoint)

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
            val[val > 0] *= abs(vmax - midpoint)
            val[val < 0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val-0.5)
            if val < 0:
                return val * abs(vmin - midpoint) + midpoint
            else:
                return val * abs(vmax - midpoint) + midpoint


def fit_semiconductor(t, data, sav_n=11, sav_deg=4, mode='sav', tr=0.4):
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import leastsq
    ger = data[..., -1].sum(2).squeeze()
    plt.subplot(121)
    plt.title('Germanium sum')
    plt.plot(t, ger[:, 0])
    plt.plot(t, ger[:, 1])
    if mode == 'sav':
        plt.plot(t, savgol_filter(ger[:, 0], sav_n, sav_deg, 0))
        plt.plot(t, savgol_filter(ger[:, 1], sav_n, sav_deg, 0))
    plt.xlim(-1, 3)
    plt.subplot(122)
    plt.title('First dervitate')
    if mode == 'sav':
        derv0 = savgol_filter(ger[:, 0], sav_n, sav_deg, 1)
        derv1 = savgol_filter(ger[:, 1], sav_n, sav_deg, 1)
    elif mode == 'gauss':
        derv0 = gaussian_filter1d(ger[:, 0], sav_n, order=1)
        derv1 = gaussian_filter1d(ger[:, 1], sav_n, order=1)
    plt.plot(t, derv0)
    plt.plot(t, derv1)
    plt.xlim(-.8, .8)
    plt.ylim(0, 700)
    plt.minorticks_on()
    plt.grid(1)

    def gaussian(p, ch, res=True):

        i, j = dv.fi(t, -tr), dv.fi(t, tr)
        w = p[0]
        A = p[1]
        x0 = p[2]
        fit = A * np.exp(-(t[i:j] - x0)**2 / (2 * w**2))
        if res:
            return fit - ch[i:j]
        else:
            return fit

    x0 = leastsq(gaussian, [.2, max(derv0), 0], derv0)
    plt.plot(
        t[dv.fi(t, -tr):dv.fi(t, tr)],
        gaussian(x0[0], 0, 0),
        '--k',
    )
    plt.text(0.05,
             0.9,
             'x$_0$ = %.2f\nFWHM = %.2f\nA = %.1f\n' %
             (x0[0][2], 2.35 * x0[0][0], x0[0][1]),
             transform=plt.gca().transAxes,
             va='top')

    x0 = leastsq(gaussian, [.2, max(derv1), 0], derv1)
    plt.plot(
        t[dv.fi(t, -tr):dv.fi(t, tr)],
        gaussian(x0[0], 1, 0),
        '--b',
    )

    plt.xlim(-.8, .8)
    plt.minorticks_on()
    plt.grid(0)
    plt.tight_layout()
    plt.text(0.5,
             0.9,
             'x$_0$ = %.2f\nFWHM = %.2f\nA = %.1f\n' %
             (x0[0][2], 2.35 * x0[0][0], x0[0][1]),
             transform=plt.gca().transAxes,
             va='top')


def stack_ax(num_rows=2, num_cols=1, height_rations=[2, 1]):
    gs = plt.GridSpec(num_rows,
                      num_cols,
                      wspace=0,
                      hspace=0,
                      height_ratios=height_rations)
    # disable ticklabels
    axes = []
    for r in range(num_rows):
        row = []
        for c in range(num_cols):
            ax = plt.subplot(gs[r, c])
            row.append(ax)
            if r != num_rows:
                ax.xaxis.tick_params(label_bottom=False)
            if c != 0:
                ax.yaxis.tick_params(label_left=False)

        axes.append(row)


def nsf(num, n=1):
    """n-Significant Figures"""
    if num > 30:
        return '%4.0f' % np.around(num, -1)
    if num > 10:
        return '%4.0f' % num
    if num > 1:
        return '%4.1f' % num
    if num < 1:
        return '%4.2f' % num


def error_string(val: float,
                 err: float,
                 valname: Optional[str] = None,
                 unit: Optional[str] = None) -> str:
    """Returns a string with the value and error with correct siginificant figures"""
    digits = np.floor(np.log10(err))
    rounded_up = np.ceil(err * 10**(-digits))
    fmt = f"{{:.{int(-digits)}f}}"
    val_str = fmt.format(val)
    err_str = fmt.format(rounded_up * 10**digits)
    s = f"{val_str} ± {err_str}"
    if valname is not None:
        s = f"{valname} = {s}"
    if unit is not None:
        s += f" {unit}"
    return s


def error_string_lmfit(param: lmfit.Parameter,
                       valname: Optional[str] = None,
                       unit: Optional[str] = None) -> str:
    """Returns a string with the value and error with correct siginificant figures"""
    return error_string(param.value, param.stderr, valname, unit)


def fig_fixed_axes(axes_shape: Tuple[int, int],
                   axes_size: Tuple[float, float],
                   padding: float = 0.3,
                   left_margin: float = 0.45,
                   bot_margin: float = 0.42,
                   hspace: float = 0.1,
                   vspace: float = 0.1,
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None,
                   **kwargs):
    """Helper funtion to generate a figure form axes sizes given in inches"""
    bots = np.arange(
        0, axes_shape[0]) * (axes_size[0] + vspace) + padding + bot_margin - vspace
    tops = bots + axes_size[0]
    lefts = np.arange(
        0, axes_shape[1]) * (axes_size[1] + hspace) + padding + left_margin - hspace
    rights = lefts + axes_size[1]

    figsize = (rights.max() + padding, tops.max() + padding)
    fig = plt.figure(figsize=figsize, **kwargs)
    tr = fig.dpi_scale_trans + fig.transFigure.inverted()

    arrs = []
    first_ax = None
    for i in range(axes_shape[0]):
        cols = []
        for j in range(axes_shape[1]):
            x0, y0 = tr.transform((lefts[j], bots[i]))
            w, h = tr.transform((axes_size[1], axes_size[0]))
            ax = fig.add_axes((x0, y0, w, h), sharex=first_ax, sharey=first_ax)
            if first_ax is None:
                first_ax = ax
            ax.tick_params(labelbottom=(i == 0), labelleft=(j == 0))
            cols.append(ax)
        arrs.append(cols)

    if ylabel:
        x, y = tr.transform((padding, (tops.max() + bots.min()) / 2))
        fig.text(x, y, ylabel, rotation=90, ha='center', va='center')

    if xlabel:
        x, y = tr.transform(((lefts.max() + rights.min()) / 2, padding))
        fig.text(x, y, xlabel, ha='center', va='center')
    return fig, np.array(arrs)[::-1, :]


def symticks(ax, linthresh=1, linstep=0.2, axis='x'):
    l, r = ax.get_xlim() if axis == 'x' else ax.get_ylim()
    axis = ax.xaxis if axis == 'x' else ax.yaxis
    m = max(l, r)
    k = min(l, r)
    major = int(np.floor(np.log10(m)))
    lin_pos = np.arange(-linthresh, 0, linstep)[1:]

    major_pos = 10**np.arange(major + 1)
    minor_pos = [np.arange(2, 10) * 10**i for i in range(major)]
    rest = np.arange(np.ceil(m / 10**major)) * 10**major
    minor_pos = np.array(minor_pos).flat

    axis.set_ticks(np.hstack((-lin_pos, lin_pos[lin_pos > k], minor_pos, rest)),
                   minor=True)
    axis.set_ticks(np.hstack((0, major_pos)))
    axis.set_major_formatter(plt.ScalarFormatter())


def lbl_axes(axs=None, pos=(-.2, -.2), fmt="(%s)", labels=None, **kwargs):
    """Labels the axes in figure

    Parameters
    ----------
    axs : List[plt.Axes], optional
        The axes to label, by default None
    pos : tuple, optional
        x, y position of the label in axis coordinates , by default (-.2, -.2)
    fmt : str, optional
        Format string, by default "(%s)"
    labels : [type], optional
        The label, by default None, resulting in a, b, c, ...
    kwargs:
        will be passed to ax.text.
    """
    if axs is None:
        axs = plt.gcf().get_axes()

    if labels is None:
        labels = string.ascii_lowercase
    text_kwargs = dict(weight='bold', fontsize='large')
    text_kwargs.update(kwargs)
    for i, a in enumerate(axs):
        a.text(pos[0], pos[1], fmt % labels[i], transform=a.transAxes, **text_kwargs)


def ci_plot(ci_dict, trace):
    """
    Plots the given CI intervals. Needs the trace output from coinfidence
    intervals. Currently assumes the CI are calculated for 1,2 and 3 sigmas.

    Parameters
    ----------
    ci_dict : dict
        Out
    trace : dict
        Trace dict
    """
    n = len(ci_dict)
    fig, ax = plt.subplots(n, 1, figsize=(1.5, n * 0.8), gridspec_kw=dict(hspace=0.5))

    for i, (pname, vals) in enumerate(ci_dict.items()):
        para_trace = trace[pname]
        idx = np.argsort(para_trace[pname])

        center = vals[len(vals) // 2][1]
        arr = np.array(vals)
        b = -.2
        x, y = trace[pname][pname][idx], 1 - trace[pname]['prob'][idx]
        u, l = arr[[0, -1], 1]

        r = (x > u) & (x < l)

        xn = np.linspace(u, l, 100)
        un, idx = np.unique(x, return_index=True)

        yn = np.interp(xn, x[idx], y[idx])
        yn = interpolate.interp1d(x[idx], y[idx], 'quadratic', fill_value=0)(xn)
        ax[i].plot(arr[[0, -1], 1], [b, b], lw=1, c='k')
        ax[i].plot(arr[[1, -2], 1], [b, b], lw=3, c='k')
        ax[i].plot(arr[[2, -3], 1], [b, b], lw=5, c='k')
        ax[i].plot(center, b, 'wx')
        ax[i].plot(x[r], y[r], 'o', ms=3, mec='None', clip_on=False)
        ax[i].fill_between(xn, 0, yn, lw=0, alpha=0.8)
        ax[i].set_ylim(-.35, 1.03)
        for n in 'top', 'left', 'right':
            ax[i].spines[n].set_visible(False)
        ax[i].yaxis.set_tick_params(left=False, labelleft=False)
        ax[i].annotate(pname, (0.05, 0.90), xycoords='axes fraction')
    fig.tight_layout()


def get_fonts() -> List[str]:
    import matplotlib.font_manager
    families = []
    try:
        fpaths = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        for i in fpaths:
            f = matplotlib.font_manager.get_font(i)
            families.append(f.family_name)
    except RuntimeError:
        families = []
    return families


def enable_style():
    plt.rcParams['figure.facecolor'] = 'w'
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['figure.figsize'] = (3.2, 2.3)
    plt.rcParams['font.size'] = 9
    s = set(('Arial', 'Helvetica')).union(set(get_fonts()))
    if len(s) > 0:
        plt.rcParams['font.family'] = list(s)
    plt.rcParams['text.hinting'] = 'either'
    plt.rcParams['savefig.pad_inches'] = 0.05
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['legend.borderaxespad'] = 0.2
    plt.rcParams['legend.columnspacing'] = 0.3
    plt.rcParams['legend.handletextpad'] = 0.2
    plt.rcParams['legend.fontsize'] = 'small'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.formatter.useoffset'] = False
