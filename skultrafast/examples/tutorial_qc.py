"""
QuickControl 
============

In this tutorial we will load files recorded with QuickControl software from
Phasetech. Before starting with the actual import, we extract the example data
into an temporary directory.
"""

# %%
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

from skultrafast.quickcontrol import QCTimeRes
from skultrafast import plot_helpers

plot_helpers.enable_style()

qc_file = QCTimeRes(tmpdir.name + '/20201029#07.info')
qc_file = QCTimeRes(r'C:\Users\tills\OneDrive\Potsdam\data\202105\20210517#47.info')
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

# %%
# How to work with the dataset please look at the other tutorials.

ds_pol.plot.spec(0.5, 3, n_average=1)


# %%
fig, ax = plt.subplots()
ds_pol.plot.trans(2160, 2100, symlog=False)
ax.set_xlim(-1, 2)
# %%

# %%

# %%
