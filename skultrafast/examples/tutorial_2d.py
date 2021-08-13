"""
Two dimensional spectra
=======================

In the example we show skultrafasts current capabilties when it comes to working
with 2D-spectra. First we need to get an example spectrum. For that, we use
the example data, which will be downloaded from figshare if necessary.
"""
# %%

import tempfile
import zipfile
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np

from skultrafast import plot_helpers
from skultrafast.quickcontrol import QC2DSpec
from skultrafast.twoD_dataset import TwoDim

from skultrafast.data_io import get_twodim_dataset

p = get_twodim_dataset()

# %%
# Lets look at the content. We are looking for info files.

infos = list(p.glob('*.info'))
infos


# %%
# Two info files are in the directory, the first contains the 1D data and the
# second the 2d data. Here we will work with the 2D data, hence we select latter
# file and open it via an quickcontrol 2dspec. We use 4 times upsampling of pump
# axis and use 10 pixel left and right to calculate and subtract a background.
#
# To create the dataset to work with form raw data we call the make_ds methods,
# which returns a `TwoDim` object to work with.

plot_helpers.enable_style()

qc_file = QC2DSpec(infos[1], bg_correct=(10, 10), upsampling=4)
ds_all = qc_file.make_ds()

# %%
# First, lets get an overview and plot a contour plot at 0.5, 1 and 3 ps.

ds_all.plot.contour(0.5, 1, 3)

# %%
# We are only interested in the region of the signal, hence we select a range.
# The methods gives us a new, smaller `TwoDim` dataset.

ds = ds_all.select_range((2140, 2180), (2120, 2180))
c, ax = ds.plot.contour(0.5, 1, 3)
ax.set_aspect(1)


# %%
# To get an normal
