"""
QuickControl 
============

In this tutorial we will load files recorded with QuickControl software from
Phasetech. Before starting with the actual import, we extract the example data
into an temporary directory.
"""

# %%
from scipy.ndimage.interpolation import shift
from skultrafast.dataset import PolTRSpec, TimeResSpec
import numpy as np
import matplotlib.pyplot as plt
import zipfile, tempfile
from pathlib import Path
from skultrafast.data_io import get_example_path
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

from skultrafast.quickcontrol import QC1DSpec, QCTimeRes
from skultrafast.dataset import PolTRSpec, TimeResSpec
from skultrafast import plot_helpers

plot_helpers.enable_style()

qc_file = QC1DSpec(tmpdir.name + '/20201029#07.info')
qc_file = QC1DSpec(r'C:\Users\tills\OneDrive\Potsdam\data\20210521#135\20210521#135.info')

# %%
# The `qc_file` object contains all the content from file in python readable
# form. We can access the info by just looking at the `info` attribute.

#qc_file.info

# %%
# The data is saved in the `par_data` and `per_data` attributes. The time-delays
# are in `qc_file.t`. The wavelengths are not saved by QC and are calculated
# from the given monochromator parameters. It may be necessary to recalculate
# the wavelengths. To get an skultrafast dataset we call:

ds_pol = qc_file.make_pol_ds()
ds_pol = ds_pol.apply_filter('uniform', (1, 2))
ds_pol = ds_pol.scale_and_shift(t_shift=-0.15)


ds_pol.perp.data[:, :-2] = ds_pol.perp.data[:, 2:]



para_bg = bg_correct(ds_pol.para, deg=2)
perp_bg = bg_correct(ds_pol.perp, deg=2)
#para_bg.plot.map(con_filter=(3, 3))

# %%

ds_bg = PolTRSpec(para_bg, perp_bg)
ds_bg.iso.plot.map(con_filter=(3,3))
# %%
# How to work with the dataset please look at the other tutorials.

ds_bg.iso.plot.spec( 0.5, 1, 20, 50, n_average=1, lw=1, add_legend=True)
plt.xlim(2100, 2180)

# %%
fig, ax = plt.subplots()
ds_bg.plot.trans(2155, 2128, symlog=1)

#ax.set_xlim(-0.5, 50)


# %%
ds_bg.iso.fit_exp([0.0, 0.01, 0.3, 50], from_t=0.3, fix_last_decay=False)
ds_bg.iso.plot.das()
# %%

# %%
ds_pol.plot.trans(2200)
plt.xlim(-10)
# %%
