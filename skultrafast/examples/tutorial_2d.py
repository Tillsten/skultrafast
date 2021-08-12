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
from skultrafast.quickcontrol import QC1DSpec, QC2DSpec, bg_correct

from skultrafast.data_io import get_twodim_dataset
import zipfile_deflate64

p = get_twodim_dataset()

# %%
# Lets look at the content. We are looking for info files.

infos = list(p.glob('*.info'))
infos


# %%
# Two info files are in the directory, the first contains

plot_helpers.enable_style()

qc_file = QC2DSpec(infos[1], bg_correct=(10, 10), upsampling=4)
ds_all = qc_file.make_ds()


# %%
ds_all.plot.contour(1)
# %%

ds = ds_all.select_range((2140, 2180), (2120, 2180))
c, ax = ds.plot.contour(1, aspect=1)
ax.set_aspect(1)

# %%
