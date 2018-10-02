# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 20:20:34 2018

@author: Tillsten
"""

import matplotlib.pyplot as plt
import numpy as np
import skultrafast.data_io
from skultrafast import zero_finding
from scipy.ndimage import gaussian_filter1d
wl, t, dat = skultrafast.data_io.load_example()
plt.rcParams['font.family'] = 'serif'

gw = {'height_ratios': (3, 2, 2)}
fig, axs = plt.subplots(3, 1, num='test', figsize=(5, 6), gridspec_kw=gw,
                        sharex='col', constrained_layout=True)

t = t*1000
y = dat[:, 200]
axs[0].plot(t, dat[:, 200], '-sk', mec='w', ms=4)
axs[0].set_xlim(-.5*1000, 1.5*1000)
axs[-1].set_xlabel('Time [fs]')
axs[0].set_ylabel('mOD')

ld = axs[1].plot(t[1:], np.diff(y), '-o', c='C0')
axs[1].set_ylabel('diff')

axs[2].plot(t, gaussian_filter1d(y, 1.5, order=1), '-o', c='C1')
axs[2].set_ylabel('gauss diff')
axs[2].text(-450, -1.2, 'sigma=1.5',)
axs[0].text(750, -.7, 'val=2', color='C4')
td = zero_finding.use_diff(dat[:, 200])
tm = np.argmax(y)
tg = zero_finding.use_gaussian(dat, sigma=1)[200]
ta = zero_finding.use_first_abs(y, val=2)

axs[0].plot(t[td], y[td], 'o', ms=10, zorder=1, c='C0', label='use_diff')

axs[0].plot(t[tg], y[tg], 'D', ms=12, zorder=.9, c='C1', label='use_gaussian')
axs[0].plot(t[tm], y[tm], '^', ms=10, zorder=1, c='C2', label='use_max')
axs[0].plot(t[ta], y[ta], '>', ms=10, zorder=.9, c='C4', label='use_first_abs')
axs[0].axhline(-2, alpha=0.5, c='C4')
axs[0].axhline(2, alpha=0.5, c='C4')
axs[0].legend()
