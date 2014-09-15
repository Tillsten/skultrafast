# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:35:22 2014

@author: tillsten
"""

import matplotlib.pyplot as plt
#plt.rcParams['savefig.dpi'] = 90
#plt.rcParams['figure.figsize'] = (6, 4)
import skultrafast.dv as dv
import numpy as np

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

tableau20 = [(r/255., g/255., b/255.) for r,g,b, in tableau20]
plt.rcParams['axes.color_cycle'] =  tableau20[::2]



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
    ax.legend(['parallel', 'perpendicular'], columnspacing=0.3, ncol=2, frameon=0)

    ax.xaxis.tick_top()
    ax.set_ylabel('$\\Delta$abs / mOD')
    ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.1,  'signal average\nfor %.1f...%.0f ps'%t_range,
            transform=ax.transAxes)
            #horizontalalignment='center')

    ax2 = plt.subplot(212, sharex=ax)
    d = pd/sd
    ang = np.arccos(np.sqrt((2*d-1)/(d+2)))/np.pi*180

    ax2.plot(wl, ang, 'o-')
    ax2.set_ylim(0, 90)
    ax2.set_ylabel('angle / degrees')
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
    ax2.set_xlabel('wavenumber / cm$^{-1}$')
    ax2.set_title('angle calculated from dichroic ratio', fontsize='x-small')
    plt.tight_layout(rect=[0, 0, 1, 1], h_pad=0)
    return ax, ax2, ax3

def make_angle_plot(wl, t, para, senk, t_range):
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
    ax.set_ylabel('$\\Delta$abs / mOD')
    ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.05,  'signal average\nfor %.1f...%.1f ps'%t_range,
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

    ax.set_xlabel('frequency / cm$^{-1}$')
    ax.set_ylabel('$\\Delta$abs  / mOD')

    ax.invert_xaxis()
    ax.axhline(0, c='k', zorder=1.5)

    plt.minorticks_on()


def lbl_trans(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_xlabel('t / ps')
    ax.set_ylabel('$\\Delta$abs  / mOD')
    ax.axhline(0, c='k', zorder=1.5)
    plt.minorticks_on()

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