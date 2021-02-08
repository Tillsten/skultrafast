# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:35:22 2014

@author: tillsten
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import skultrafast.dv as dv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, SymLogNorm
import matplotlib.cbook as cbook
from scipy import interpolate
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
inv_freq = False
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
    if unit is 'nm':
        sub_axis.set_label('Wavelengths [nm]')
    elif unit is 'cm':
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
    #horizontalalignment='center')

    ax2 = plt.subplot(212, sharex=ax)
    d = pd / sd
    ang = np.arccos(np.sqrt((2*d - 1) / (d+2))) / np.pi * 180

    ax2.plot(wl, ang, 'o-')
    ax2.set_ylim(0, 90)
    ax2.set_ylabel('Angle / Degrees')
    ax3 = plt.twinx()
    ax3.plot(wl, ang, lw=0)
    ax2.invert_xaxis()
    f = lambda x: "%.1f" % (to_dichro(float(x) / 180. * np.pi))
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
    #ax.xaxis.tick_top()
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
    #horizontalalignment='center')

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
    #plt.minorticks_on()


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
    fr_cm = -dv.fs2cm(1000 / freqs[n:])

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

            #First scale to -1 to 1 range, than to from 0 to 1.
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
    #disable ticklabels
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


def plot_das(fitter):
    pass


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


import string


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
        axs  = plt.gcf().get_axes()

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
    fig, ax = plt.subplots(n, 1, figsize=(1.5, n*0.8), gridspec_kw=dict(hspace=0.5))
    
    for i, (pname, vals) in enumerate(ci_dict.items()):
        para_trace = trace[pname] 
        idx = np.argsort(para_trace[pname])
        
        center = vals[len(vals)//2][1]
        arr = np.array(vals)
        b = -.2
        x, y = trace[pname][pname][idx], 1-trace[pname]['prob'][idx]
        u, l = arr[[0, -1], 1]
        
        r = (x > u) & (x < l)
        
        xn = np.linspace(u, l, 100)
        un, idx = np.unique(x, return_index=True)
        
        yn = np.interp(xn, x[idx], y[idx])
        yn = interpolate.interp1d(x[idx], y[idx], 'quadratic',
                                  fill_value=0)(xn)
        ax[i].plot(arr[[0, -1], 1], [b, b], lw=1, c='k')
        ax[i].plot(arr[[1, -2], 1], [b, b], lw=3, c='k')
        ax[i].plot(arr[[2, -3], 1], [b, b], lw=5, c='k')
        ax[i].plot(center,  b, 'wx' )
        ax[i].plot(x[r], y[r], 'o', ms=3, mec='None', clip_on=False)
        ax[i].fill_between(xn, 0, yn, lw=0, alpha=0.8)
        ax[i].set_ylim(-.35, 1.03)
        for n in 'top', 'left', 'right':
            ax[i].spines[n].set_visible(False)
        ax[i].yaxis.set_tick_params(left=False, labelleft=False)
        ax[i].annotate(pname, (0.05, 0.90), xycoords='axes fraction')
    fig.tight_layout()


def enable_style():
    plt.rcParams['figure.facecolor'] = 'w'
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['figure.figsize'] = (3.2, 2.3)
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.sans-serif'] = 'Arial'
    # plt.rcParams['font.serif'] = 'Helvetica'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.hinting'] = 'either'
    plt.rcParams['savefig.pad_inches'] = 0.05
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['legend.borderaxespad'] = 0.2
    plt.rcParams['legend.columnspacing'] = 0.3
    plt.rcParams['legend.handletextpad'] = 0.2
    plt.rcParams['legend.fontsize'] = 'small'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.formatter.useoffset'] = False


from matplotlib.colors import LinearSegmentedColormap

cmaps = {}

cm_data = [[0.26710521, 0.03311059, 0.6188155], [0.26493929, 0.04780926, 0.62261795],
           [0.26260545, 0.06084214, 0.62619176], [0.26009691, 0.07264411, 0.62951561],
           [0.25740785, 0.08360391, 0.63256745], [0.25453369, 0.09395358, 0.63532497],
           [0.25147146, 0.10384228, 0.6377661], [0.24822014, 0.11337029, 0.6398697],
           [0.24478105, 0.12260661, 0.64161629], [0.24115816, 0.131599, 0.6429888],
           [0.23735836, 0.14038009, 0.64397346], [0.23339166, 0.14897137, 0.64456048],
           [0.22927127, 0.15738602, 0.64474476], [0.22501278, 0.16563165, 0.64452595],
           [0.22063349, 0.17371215, 0.64390834], [0.21616055, 0.18162302, 0.64290515],
           [0.21161851, 0.18936156, 0.64153295], [0.20703353, 0.19692415, 0.63981287],
           [0.20243273, 0.20430706, 0.63776986], [0.19784363, 0.211507, 0.63543183],
           [0.19329361, 0.21852157, 0.63282872], [0.18880937, 0.2253495, 0.62999156],
           [0.18442119, 0.23198815, 0.62695569], [0.18014936, 0.23844124, 0.62374886],
           [0.17601569, 0.24471172, 0.62040016], [0.17204028, 0.25080356, 0.61693715],
           [0.16824123, 0.25672163, 0.6133854], [0.16463462, 0.26247158, 0.60976836],
           [0.16123449, 0.26805963, 0.60610723], [0.15805279, 0.27349243, 0.60242099],
           [0.15509948, 0.27877688, 0.59872645], [0.15238249, 0.28392004, 0.59503836],
           [0.14990781, 0.28892902, 0.59136956], [0.14767951, 0.29381086, 0.58773113],
           [0.14569979, 0.29857245, 0.58413255], [0.1439691, 0.30322055, 0.58058191],
           [0.14248613, 0.30776167, 0.57708599], [0.14124797, 0.31220208, 0.57365049],
           [0.14025018, 0.31654779, 0.57028011], [0.13948691, 0.32080454, 0.5669787],
           [0.13895174, 0.32497744, 0.56375063], [0.13863958, 0.32907012, 0.56060453],
           [0.138537, 0.3330895, 0.55753513], [0.13863384, 0.33704026, 0.55454374],
           [0.13891931, 0.34092684, 0.55163126], [0.13938212, 0.34475344, 0.54879827],
           [0.14001061, 0.34852402, 0.54604503], [0.14079292, 0.35224233, 0.54337156],
           [0.14172091, 0.35590982, 0.54078769], [0.14277848, 0.35953205, 0.53828312],
           [0.14395358, 0.36311234, 0.53585661], [0.1452346, 0.36665374, 0.5335074],
           [0.14661019, 0.3701591, 0.5312346], [0.14807104, 0.37363011, 0.52904278],
           [0.1496059, 0.3770697, 0.52692951], [0.15120289, 0.3804813, 0.52488853],
           [0.15285214, 0.38386729, 0.52291854], [0.15454421, 0.38722991, 0.52101815],
           [0.15627225, 0.39056998, 0.5191937], [0.15802555, 0.39389087, 0.5174364],
           [0.15979549, 0.39719482, 0.51574311], [0.16157425, 0.40048375, 0.51411214],
           [0.16335571, 0.40375871, 0.51254622], [0.16513234, 0.40702178, 0.51104174],
           [0.1668964, 0.41027528, 0.50959299], [0.16864151, 0.41352084, 0.50819797],
           [0.17036277, 0.41675941, 0.50685814], [0.1720542, 0.41999269, 0.50557008],
           [0.17370932, 0.42322271, 0.50432818], [0.17532301, 0.42645082, 0.50313007],
           [0.17689176, 0.42967776, 0.50197686], [0.17841013, 0.43290523, 0.5008633],
           [0.17987314, 0.43613477, 0.49978492], [0.18127676, 0.43936752, 0.49873901],
           [0.18261885, 0.44260392, 0.49772638], [0.18389409, 0.44584578, 0.49673978],
           [0.18509911, 0.44909409, 0.49577605], [0.18623135, 0.4523496, 0.494833],
           [0.18728844, 0.45561305, 0.49390803], [0.18826671, 0.45888565, 0.49299567],
           [0.18916393, 0.46216809, 0.49209268], [0.18997879, 0.46546084, 0.49119678],
           [0.19070881, 0.46876472, 0.49030328], [0.19135221, 0.47208035, 0.48940827],
           [0.19190791, 0.47540815, 0.48850845], [0.19237491, 0.47874852, 0.4876002],
           [0.19275204, 0.48210192, 0.48667935], [0.19303899, 0.48546858, 0.48574251],
           [0.19323526, 0.48884877, 0.48478573], [0.19334062, 0.49224271, 0.48380506],
           [0.19335574, 0.49565037, 0.4827974], [0.19328143, 0.49907173, 0.48175948],
           [0.19311664, 0.50250719, 0.48068559], [0.192864, 0.50595628, 0.47957408],
           [0.19252521, 0.50941877, 0.47842186], [0.19210087, 0.51289469, 0.47722441],
           [0.19159194, 0.516384, 0.47597744], [0.19100267, 0.51988593, 0.47467988],
           [0.19033595, 0.52340005, 0.47332894], [0.18959113, 0.5269267, 0.47191795],
           [0.18877336, 0.530465, 0.47044603], [0.18788765, 0.53401416, 0.46891178],
           [0.18693822, 0.53757359, 0.46731272], [0.18592276, 0.54114404, 0.46563962],
           [0.18485204, 0.54472367, 0.46389595], [0.18373148, 0.5483118, 0.46207951],
           [0.18256585, 0.55190791, 0.4601871], [0.18135481, 0.55551253, 0.45821002],
           [0.18011172, 0.55912361, 0.45615277], [0.17884392, 0.56274038, 0.45401341],
           [0.17755858, 0.56636217, 0.45178933], [0.17625543, 0.56998972, 0.44946971],
           [0.174952, 0.57362064, 0.44706119], [0.17365805, 0.57725408, 0.44456198],
           [0.17238403, 0.58088916, 0.4419703], [0.17113321, 0.58452637, 0.43927576],
           [0.1699221, 0.58816399, 0.43648119], [0.1687662, 0.5918006, 0.43358772],
           [0.16767908, 0.59543526, 0.43059358], [0.16667511, 0.59906699, 0.42749697],
           [0.16575939, 0.60269653, 0.42428344], [0.16495764, 0.6063212, 0.42096245],
           [0.16428695, 0.60993988, 0.41753246], [0.16376481, 0.61355147, 0.41399151],
           [0.16340924, 0.61715487, 0.41033757], [0.16323549, 0.62074951, 0.40656329],
           [0.16326148, 0.62433443, 0.40266378], [0.16351136, 0.62790748, 0.39864431],
           [0.16400433, 0.63146734, 0.39450263], [0.16475937, 0.63501264, 0.39023638],
           [0.16579502, 0.63854196, 0.38584309], [0.16712921, 0.64205381, 0.38132023],
           [0.168779, 0.64554661, 0.37666513], [0.17075915, 0.64901912, 0.37186962],
           [0.17308572, 0.65246934, 0.36693299], [0.1757732, 0.65589512, 0.36185643],
           [0.17883344, 0.65929449, 0.3566372], [0.18227669, 0.66266536, 0.35127251],
           [0.18611159, 0.66600553, 0.34575959], [0.19034516, 0.66931265, 0.34009571],
           [0.19498285, 0.67258423, 0.3342782], [0.20002863, 0.67581761, 0.32830456],
           [0.20548509, 0.67900997, 0.3221725], [0.21135348, 0.68215834, 0.31587999],
           [0.2176339, 0.68525954, 0.30942543], [0.22432532, 0.68831023, 0.30280771],
           [0.23142568, 0.69130688, 0.29602636], [0.23893914, 0.69424565, 0.28906643],
           [0.2468574, 0.69712255, 0.28194103], [0.25517514, 0.69993351, 0.27465372],
           [0.26388625, 0.70267437, 0.26720869], [0.27298333, 0.70534087, 0.25961196],
           [0.28246016, 0.70792854, 0.25186761], [0.29232159, 0.71043184, 0.2439642],
           [0.30253943, 0.71284765, 0.23594089], [0.31309875, 0.71517209, 0.22781515],
           [0.32399522, 0.71740028, 0.21959115], [0.33520729, 0.71952906, 0.21129816],
           [0.3467003, 0.72155723, 0.20298257], [0.35846225, 0.72348143, 0.19466318],
           [0.3704552, 0.72530195, 0.18639333], [0.38264126, 0.72702007, 0.17822762],
           [0.39499483, 0.72863609, 0.17020921], [0.40746591, 0.73015499, 0.1624122],
           [0.42001969, 0.73158058, 0.15489659], [0.43261504, 0.73291878, 0.14773267],
           [0.44521378, 0.73417623, 0.14099043], [0.45777768, 0.73536072, 0.13474173],
           [0.47028295, 0.73647823, 0.1290455], [0.48268544, 0.73753985, 0.12397794],
           [0.49497773, 0.73854983, 0.11957878], [0.5071369, 0.73951621, 0.11589589],
           [0.51913764, 0.74044827, 0.11296861], [0.53098624, 0.74134823, 0.11080237],
           [0.5426701, 0.74222288, 0.10940411], [0.55417235, 0.74308049, 0.10876749],
           [0.56550904, 0.74392086, 0.10885609], [0.57667994, 0.74474781, 0.10963233],
           [0.58767906, 0.74556676, 0.11105089], [0.59850723, 0.74638125, 0.1130567],
           [0.609179, 0.74719067, 0.11558918], [0.61969877, 0.74799703, 0.11859042],
           [0.63007148, 0.74880206, 0.12200388], [0.64030249, 0.74960714, 0.12577596],
           [0.65038997, 0.75041586, 0.12985641], [0.66034774, 0.75122659, 0.1342004],
           [0.67018264, 0.75203968, 0.13876817], [0.67990043, 0.75285567, 0.14352456],
           [0.68950682, 0.75367492, 0.14843886], [0.69900745, 0.75449768, 0.15348445],
           [0.70840781, 0.75532408, 0.15863839], [0.71771325, 0.75615416, 0.16388098],
           [0.72692898, 0.75698787, 0.1691954], [0.73606001, 0.75782508, 0.17456729],
           [0.74511119, 0.75866562, 0.17998443], [0.75408719, 0.75950924, 0.18543644],
           [0.76299247, 0.76035568, 0.19091446], [0.77183123, 0.76120466, 0.19641095],
           [0.78060815, 0.76205561, 0.20191973], [0.78932717, 0.76290815, 0.20743538],
           [0.79799213, 0.76376186, 0.21295324], [0.8066067, 0.76461631, 0.21846931],
           [0.81517444, 0.76547101, 0.22398014], [0.82369877, 0.76632547, 0.2294827],
           [0.832183, 0.7671792, 0.2349743], [0.8406303, 0.76803167, 0.24045248],
           [0.84904371, 0.76888236, 0.24591492], [0.85742615, 0.76973076, 0.25135935],
           [0.86578037, 0.77057636, 0.25678342], [0.87410891, 0.77141875, 0.2621846],
           [0.88241406, 0.77225757, 0.26755999], [0.89070781, 0.77308772, 0.27291122],
           [0.89898836, 0.77391069, 0.27823228], [0.90725475, 0.77472764, 0.28351668],
           [0.91550775, 0.77553893, 0.28875751], [0.92375722, 0.7763404, 0.29395046],
           [0.9320227, 0.77712286, 0.29909267], [0.94027715, 0.7779011, 0.30415428],
           [0.94856742, 0.77865213, 0.3091325], [0.95686038, 0.7793949, 0.31397459],
           [0.965222, 0.7800975, 0.31864342], [0.97365189, 0.78076521, 0.32301107],
           [0.98227405, 0.78134549, 0.32678728], [0.99136564, 0.78176999, 0.3281624],
           [0.99505988, 0.78542889, 0.32106514], [0.99594185, 0.79046888, 0.31648808],
           [0.99646635, 0.79566972, 0.31244662], [0.99681528, 0.80094905, 0.30858532],
           [0.9970578, 0.80627441, 0.30479247], [0.99724883, 0.81161757, 0.30105328],
           [0.99736711, 0.81699344, 0.29725528], [0.99742254, 0.82239736, 0.29337235],
           [0.99744736, 0.82781159, 0.28943391], [0.99744951, 0.83323244, 0.28543062],
           [0.9973953, 0.83867931, 0.2812767], [0.99727248, 0.84415897, 0.27692897],
           [0.99713953, 0.84963903, 0.27248698], [0.99698641, 0.85512544, 0.26791703],
           [0.99673736, 0.86065927, 0.26304767], [0.99652358, 0.86616957, 0.25813608],
           [0.99622774, 0.87171946, 0.25292044], [0.99590494, 0.87727931, 0.24750009],
           [0.99555225, 0.88285068, 0.2418514], [0.99513763, 0.8884501, 0.23588062],
           [0.99471252, 0.89405076, 0.2296837], [0.99421873, 0.89968246, 0.2230963],
           [0.99370185, 0.90532165, 0.21619768], [0.99313786, 0.91098038, 0.2088926],
           [0.99250707, 0.91666811, 0.20108214], [0.99187888, 0.92235023, 0.19290417],
           [0.99110991, 0.92809686, 0.18387963], [0.99042108, 0.93379995, 0.17458127],
           [0.98958484, 0.93956962, 0.16420166], [0.98873988, 0.94533859, 0.15303117],
           [0.98784836, 0.95112482, 0.14074826], [0.98680727, 0.95697596, 0.12661626]]

parula = LinearSegmentedColormap.from_list(__file__, cm_data)
