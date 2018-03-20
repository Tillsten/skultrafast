# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 21:33:24 2015

@author: Tillsten
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

tableau20 = [(r/255., g/255., b/255.) for r,g,b, in tableau20]

#plt.rcParams['savefig.dpi'] = 110
#plt.rcParams['font.family'] = 'Vera Sans'

out_ticks = {'xtick.direction': 'out',
             'xtick.major.width': 1.5,
             'xtick.minor.width': 1,
             'xtick.major.size': 6,
             'xtick.minor.size': 3,
             'xtick.minor.visible': True,
             'ytick.direction': 'out',
             'ytick.major.width': 1.5,
             'ytick.minor.width': 1,
             'ytick.major.size': 6,
             'ytick.minor.size': 3,
             'ytick.minor.visible': True,
             'axes.spines.top': False,
             'axes.spines.right': False,
             'text.hinting': True,
             'axes.titlesize': 'xx-large',
             'axes.titleweight': 'semibold',
             }


plt.figure(figsize=(6,4))

with plt.style.context(out_ticks):
    ax = plt.subplot(111)
    x = np.linspace(0, 7, 1000)
    y = np.exp(-x/1.5)*np.cos(x/1*(2*np.pi))#*np.cos(x/0.05*(2*np.pi))
    l, = plt.plot(x, np.exp(-x/1.5), lw=0.5, color='grey')
    l, = plt.plot(x, -np.exp(-x/1.5), lw=0.5, color='grey')
    l, = plt.plot(x, y, lw=1.1)
    #l.set_clip_on(0)
    plt.tick_params(which='both', top=False, right=False)
    plt.margins(0.01)
    ax.text(7, 1, r'$y(t)=\exp\left(-t/1.5\right)\cos(\omega_1t)\cos(\omega_2t)$',
             fontsize=18, va='top', ha='right')
    #plt.title("Hallo")
    plt.setp(plt.gca(), xlabel='Time [s]', ylabel='Amplitude')
    ax = plt.axes([0.57, 0.25, 0.3, .2])
    #ax.plot(np.fft.fftfreq(x.size)[:y.size/2], abs(np.fft.fft(y))[:y.size/2])
    ax.fill_between(np.fft.fftfreq(x.size, x[1]-x[0])[:y.size/2],
                    abs(np.fft.fft(y))[:y.size/2], alpha=0.2, color='r')
    ax.set_xlim(0, 10)
    ax.set_xlabel("Frequency")
    ax.xaxis.labelpad = 1

    plt.locator_params(nbins=4)
    plt.tick_params(which='both', top=False, right=False)
    plt.tick_params(which='minor', bottom=False, left=False)

    #plt.grid(1, axis='y', linestyle='-', alpha=0.3, lw=.5)
plt.show()
