"""
QuickControl
============

In this tutorial we will load files recorded with QuickControl software from
Phasetech. Before starting with the actual import, we extract the example data
into an temporary directory.

Weill st
"""
# %%
#
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from skultrafast.data_io import get_example_path

p = get_example_path('quickcontrol')
tmpdir = tempfile.TemporaryDirectory()
zipfile.ZipFile(p).extractall(tmpdir.name)

# %%
# Lets look at the content.

list(Path(tmpdir.name).glob('*.*'))

# %%
# 2D Dataset
# ----------
# Now we will see how to load a 2D dataset. Again, the info file
# is the entry point.

pstr = r'D:\boxup\AG Mueller-Werkmeister\Sweet Project\2D-IR raw data\2DIR-Daten\MeSCN'
p = Path(pstr)
fname = list(p.glob('**/*320.info'))[0]
print(fname)
two_d = QC2DSpec(fname=fname, upsampling=4, bg_correct=(30, 30), probe_filter=2)
two_dim_ds = two_d.make_ds()

# %%
# First we select the range containing the data and show the spectrum
# at 0, 0.5, 1 and 4 ps.
ds = two_dim_ds.select_range((2130, 2180), (2115, 2185))
ds.plot.contour(0, 0.5, 1, 4)

# %%
# Lets calculate the the raw center line slope at 5 ps
# and plot it into the contour plot.
x, y, r = ds.single_cls(0, pr_range=10, pu_range=7)

import proplot

fig, ax = proplot.subplots(dpi=120, axwidth="4cm")
ds.plot.contour(0.1, ax=ax)
ax.plot(y, x, lw=0, c='r', marker='o', ms=3)
m, n = r.slope, r.intercept

ax.plot([m * x.min() + n, m * x.max() + n], [x.min(), x.max()], c='w', lw=1)

# %%
# There is also an method to calculate the cls for every time point.
cls_result = ds.cls(pr_range=10, pu_range=7, mode='neg')

# We can directly fit the resulting cls-decay with an exponential model
fr = cls_result.exp_fit([2])

# %%
# Lets plot the result and look at the paramenters.
fig, ax = plt.subplots()

ax.errorbar(fr.userkws['x'],
            cls_result.slopes,
            cls_result.slope_errors,
            lw=0,
            elinewidth=1,
            marker='o',
            ms=2)
ax.plot(fr.userkws['x'], fr.best_fit, c='red', lw=1)
ax.set_xscale('log')

fr

# %%
ds.plot.pump_slice_amps()
# %%
x, y = ds.pump_wn, ds.spec2d[2].ptp(axis=0).T
#plt.plot(x, y)
import lmfit

mod = (lmfit.models.GaussianModel() + lmfit.models.PolynomialModel(0))
res = mod.fit(y, x=x, center=2160, b_center=2150, c0=0)

xu = np.linspace(ds.pump_wn.min(), ds.pump_wn.max())
plt.plot(x, y, marker='s', ms=2, lw=0)
plt.plot(xu, res.eval(x=xu))

for c in res.eval_components(x=xu).values():
    plt.plot(xu, c, lw=0.5)
#
# %%
res
# %%
