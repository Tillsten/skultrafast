"""
2D Spectroscopy
===============

In this example we show skultrafasts current capabilties when it comes to
working with 2D-spectra. First we need to get a sample spectrum. For that, we
use example data, which will be downloaded from figshare if necessary. The data
was measured with the quickcontrol-software from phasetech.

We start by import the required functionalty.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from skultrafast import plot_helpers
from skultrafast import data_io
from skultrafast.quickcontrol import QC2DSpec
from skultrafast.twoD_dataset import TwoDim

from skultrafast.data_io import get_twodim_dataset

# %%
# The following line returns a path to a folder containing the sample data. If
# necessary, it will try to download the data from fighare.

p = get_twodim_dataset()

# %%
# Lets look at the content of the folder. For measurements with quickcontrol, we
# are looking for `.info` files which contain all necessary information.

infos = list(p.glob('*.info'))
infos

# %%
# Loading the data
# ----------------
#
# There are two `.info`-files the directory. The first contains the transient
# 1D-data and the second the 2d-data. Here in this tutorial we will work with
# the 2D data. Hence we select the second file and open it by instancing an
# `QC2DSpec`-class. Given the info-file, the class collects all necessary data.
# It is also responsible to turn the saved data, which are inferogramms, into
# 2D-spectra. This process can also include some preprocessing. Here we we apply
# 4 times upsampling of pump axis and use 10 pixel left and right to calculate
# and subtract a background. For apodization we are use the default hamming window.

plot_helpers.enable_style()

qc_file = QC2DSpec(infos[1], bg_correct=(10, 10), upsampling=4)

# %%
# To create the dataset to work with form raw data we call the make_ds methods,
# which returns a dict of `TwoDim` objects to work with. The dict contains
# parallel (`para`), perpendicular (`perp`) and isotropic (`iso`) datasets. We
# select the isotropic dataset.

ds_all = qc_file.make_ds()
ds_iso = ds_all['iso']

# %%
# The `TwoDim`-objects are the core structure to work with. They contain both
# frequency-axis, `pump_freqs` and `probe_freqs`, the waiting times `t`, and the
# actual data `spec2d`.
#
# To start and to get an overview, we plot a contour-plot at 1 ps. Like for the
# `TimeResSpec`-class, plotting functions are accessible under the plot
# attribute.

ds_iso.plot.contour(1, aspect=1)

# %%
# Selecting a subrange
# --------------------
#
# Most of the 2D-map is empty. We are only interested in the region of the
# signal, hence we have to select a sub-range. The methods return a new and
# smaller `TwoDim` dataset. In general, most methods which would modify
# the data return a new dataset. Now we also plot the contour at different
# time-points, 0.5, 1 and 7 ps.

ds = ds_iso.select_range((2140, 2180), (2120, 2180))
c, _ = ds.plot.contour(0.5, 1, 7, aspect=1, direction='h')

# %%
# There is also a `.select_t_range` method.
#
# Extracting a transient 1D-spectrum
# ----------------------------------
#
# Using the projection theorem and integrating over the pump axis, we can
# get a normal transient 1D-dataset. This can be done by the
# `TwoDim.integrate_method`, which returns a skultrafast `TimeResSpec`. Here we
# integrate over the whole range. It is possible to integrate over a sub-range
# by supplying arguments to the function.

ds1d = ds.integrate_pump()

fig, (ax0, ax1) = plt.subplots(2, figsize=(3, 4))
ds1d.plot.spec(0.5, 1, 7, add_legend=True, ax=ax0)
ds1d.plot.trans(2160, 2135, add_legend=True, ax=ax1)


# %%
# Center line slope analysis
# --------------------------
#
# One of the most common ways to analyze a two-dimensional dataset is to
# extract the frequency-frequency correlation function (FFCF). The most common
# ways is to determine the center line slope, which under certain assumptions is
# proportional to the normalized FFCF.
#
# To extract the cls for a single time-point, we use the `single_cls`-method.
# Let's determine the cls for 1 ps. We use a window of 10 cm-1 in both pump and
# probe axis around the maximum for determination. The algorithm currently uses
# the center-of-mass.

# sphinx_gallery_thumbnail_number = 4
y_cls, x_cls, lin_fit = ds.single_cls(1, pr_range=10, pu_range=10)

# Plot the result
_, ax = ds.plot.contour(1, aspect=1)
ax = ax[0]

# First plot the maxima
ax.plot(x_cls, y_cls, color='yellow', marker='o', markersize=3, lw=0)

# Plot the resulting fit. Since the slope is a function of the pump frequencies,
# we have to use y-values for the slope.
ax.plot(y_cls * lin_fit.slope + lin_fit.intercept, y_cls, color='w')

# %%
# To determine the full CLS-decay, we can use the `cls`-method. It takes the
# same arguments as `single_cls`, except the single waiting time. The
# information of each single cls is accessible in the cls result.

cls_result = ds.cls(pr_range=10, pu_range=10)

ti = ds.t_idx(1)
_, ax = ds.plot.contour(1, aspect=1)
ax = ax[0]
x_cls, y_cls = cls_result.lines[ti][:, 1], cls_result.lines[ti][:, 0]
ax.plot(x_cls, y_cls,
        marker='o', markersize=3, lw=0, color='yellow', )
ax.plot(cls_result.slopes[ti] * y_cls + cls_result.intercepts[ti], y_cls, c='w')

# %%
# Lets look at the time-dependence of the slope.

fig, ax = plt.subplots()
ax.plot(cls_result.wt, cls_result.slopes)
ax.set(xlabel='Waiting Time', ylabel='Slope')

# %%
# The ClsResult class also offers a convenience function to the fit cls with
# exponential functions.

tau_estimate = [5.]
fr = cls_result.exp_fit(tau_estimate, use_const=True, use_weights=True)

fig, ax = plt.subplots()
cls_result.plot_cls(ax=ax)
text = '(%.1f ± %.1f) ps' % (fr.params['a_decay'].value, fr.params['a_decay'].stderr)
ax.annotate(text, (0.98, 0.98), xycoords='axes fraction', ha='right', va='top',
            fontsize='large')
ax.set_xscale('log')

# %%
# Notice that there a multiple methods to calculate the extrema position for
# each pump wavelengths. skultrafast currently supports center-of-mass,
# quadratic fit, quadratic fit of the log-vales and gaussian fit methods. In
# general, the results of all methods should not differ much. This is shown
# below.

fig, ax = plt.subplots()
methods = 'log_quad', 'quad', 'fit', 'com'
for m in methods:
    cls_result_fit = ds.cls(pr_range=12, pu_range=10, method=m)
    fr_fit = cls_result_fit.exp_fit(tau_estimate,  use_const=True, use_weights=True)
    data_line, _ = cls_result_fit.plot_cls(ax=ax, symlog=True)
    data_line.set_label(m)

ax.legend()

# %%
# What is often more problematic is the sensitivity to the chosen region.
# It is often suggested taking only a small region around the peak, but this
# makes the determination of the slope more error-prone. In most settings,
# the resolution of the pump axis is rather limited.

fig, ax = plt.subplots()
pump_range = 5, 7, 10, 12,
for r in pump_range:
    cls_result_fit = ds.cls(pr_range=r, pu_range=r, method='com')

    cls_result_fit.plot_cls(ax=ax, symlog=True)

ax.legend()

# %%
# Diagonal
# --------
#
# Another common usage of two-dimensional spectra is the extraction of the
# diagonal. The signal on diagonal is proportional to the fourth power of the
# transition dipole-moment, in contrast to the regular absorption
# (e.g. FTIR) spectrum, which is  proportional to the second power. Hence, shoulders
# of peaks are often more distinct in the diagonal. The diagonal can be extracted via the
# 'diag_and_antidag'-method. The methods returns an object containing both the
# diagonal and anti-diagonal, additional it contains the y-coordinates of the
# lines. Latter can be useful for plotting. Again the method takes a waiting time
# to select the spectrum for extraction. Additionally, it takes parameters to shift
# the position of the diagonal used for extraction. If not given, it goes through
# the position of the minimum.

diag_result = ds.diag_and_antidiag(0.2)

fig, ax = plt.subplots()
ax.plot(ds.probe_wn, diag_result.diag)
ax.plot(ds.probe_wn, diag_result.antidiag)
plot_helpers.lbl_spec(ax)

# %%
# The additional attributes of the result-object can be used to draw lines in the spectrum.

fig, ax = plt.subplot_mosaic('AABB', figsize=(5, 2), constrained_layout=True)

pm = ax['A'].pcolormesh(ds.probe_wn, ds.pump_wn, ds.spec2d[ds.t_idx(1), :, :].T,
              shading='auto', cmap='seismic')
ax['A'].plot(ds.probe_wn, diag_result.diag_coords, lw=1, c="y", ls='--')
ax['A'].plot(ds.probe_wn, diag_result.antidiag_coords, lw=1, c="c", ls='--')
ax['A'].set(ylim=(ds.pump_wn.min(), ds.pump_wn.max()), ylabel=plot_helpers.freq_label)
ax['A'].set_xlabel(plot_helpers.freq_label)
fig.colorbar(pm, ax=ax['A'], shrink=0.69, pad=0)

ax['B'].plot(ds.probe_wn, diag_result.diag, c='y')
ax['B'].plot(ds.probe_wn, diag_result.antidiag, c='c')
ax['B'].set_xlabel(plot_helpers.freq_label)

fig.align_xlabels()
# %%
# As an alternative, the pump-slice-amplitude method is also supported. Here we are calculating the
# difference between the maximum and minimum signal along the probe axis for each pump frequency.
# The resulting curve shares the same properties as the diagonal but is supposedly less influenced
# by excited state overlap (Valentine et al. doi:10.1021/acs.jpca.1c04558)
