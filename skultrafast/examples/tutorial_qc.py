"""
QuickControl 
============

In this tutorial we will load files recorded with QuickControl software from
Phasetech. Before starting with the actual import, we extract the example data
into an temporary directory.
"""
# %%
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage.interpolation import shift
from skultrafast.data_io import get_example_path
from skultrafast.dataset import PolTRSpec, PolTRSpecPlotter, TimeResSpec

p = get_example_path('quickcontrol')
tmpdir = tempfile.TemporaryDirectory()
zipfile.ZipFile(p).extractall(tmpdir.name)

# %%
# Lets look at the content.

list(Path(tmpdir.name).glob('*.*'))

# %%
# All datasets from QC contain a `.info` file, which contains the meta
# information and the type of experiment. All skultrafast methods take the this
# file as the starting point, since all other file names can be infered from it.
#
# Lets load the info file for a 1D transient absorption experiment.

from skultrafast import plot_helpers, twoD_dataset
from skultrafast.dataset import PolTRSpec, TimeResSpec
from skultrafast.quickcontrol import QC1DSpec, QC2DSpec, bg_correct

plot_helpers.enable_style()

qc_file = QC1DSpec(tmpdir.name + '/20201029#07.info')
qc_file = QC1DSpec(r'C:\Users\tills\OneDrive\Potsdam\data\20210521#135\20210521#135.info')

# %%
# The `qc_file` object contains all the content from file in python readable
# form. We can access the info by just looking at the `info` attribute.

qc_file.info

# %%
# The data is saved in the `par_data` and `per_data` attributes. The time-delays
# are in `qc_file.t`. The wavelengths are not saved by QC and are calculated
# from the given monochromator parameters. It may be necessary to recalculate
# the wavelengths. To get an skultrafast dataset we call:

ds_pol = qc_file.make_pol_ds()
ds_pol = ds_pol.apply_filter('uniform', (1, 2))
ds_pol = ds_pol.scale_and_shift(t_shift=-0.15)


ds_pol.perp.data[:, :-2] = ds_pol.perp.data[:, 2:]



bg_correct(ds_pol.wn, ds_pol.para.data, deg=2)
bg_correct(ds_pol.wn, ds_pol.perp.data, deg=2)
bg_correct(ds_pol.wn, ds_pol.iso.data, deg=2)
#para_bg.plot.map(con_filter=(3, 3))

# %%

ds_pol.iso.plot.map(con_filter=(3,3))
# %%
# How to work with the dataset please look at the other tutorials.

ds_pol.plot.spec( 0.5, 1, 20, 50, n_average=1,  add_legend=True)
plt.xlim(2100, 2180)

# %%
# 2D Dataset
# ----------
# Now we will see how to load a 2D dataset. Again, the info file
# is the entry point.

pstr = r'D:\boxup\AG Mueller-Werkmeister\Sweet Project\2D-IR raw data\2DIR-Daten\SCN7'
p = Path(pstr)
fname =list(p.glob('**/*236.info'))[0]
two_d = QC2DSpec(fname=fname)


# %%

two_dim_ds = two_d.make_ds()
# %%
ds.plot.contour(5)
x, y, r = two_dim_ds.single_cls(5, 10, 10)
plt.plot(y, x, lw=1, c='r')
# %%
#two_dim_ds.plot.contour(5.5, region=(2200, 2100))
# %%
ds = two_dim_ds.select_range((2130, 2180), (2100, 2200))
# %%

# %%
ds.single_cls(2, 4, 4)
# %%
res = two_dim_ds.cls(pr_range=10, pu_range=10)
fr = res.exp_fit([1])
# %%
fr
# %%
fig, ax = plt.subplots()
ax.errorbar(fr.userkws['x'], res.slopes,
 res.slope_errors, lw=0, elinewidth=1, marker='o', ms=2)
ax.plot(fr.userkws['x'], fr.best_fit, c='k', lw=1)

# %%
fr
# %%
