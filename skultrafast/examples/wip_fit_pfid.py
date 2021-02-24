"""
Fitting the perturbed free induction decay
==========================================

Sometimes it is useful to fit the perturbed free induction (see explanation in
the PFID tutorial), maybe certain excited state features are not yet visible or
a more exact determination of the center frequency is required.


WORK IN PROGRESS
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import skultrafast.plot_helpers as ph
import skultrafast.unit_conversions as uc
from skultrafast.utils import pfid_r4, pfid_r6
from skultrafast import messpy, data_io, dataset
# %%


fname = data_io.get_example_path('messpy')
print("Tutorial MessPy-file located at %s" % fname)
mpf = messpy.MessPyFile(
    fname,
    invert_data=True,  # Changes the sign of the data
    is_pol_resolved=True,  # If the data was recored polarization resolved.
    pol_first_scan='perp',  # Polarisation of the first scan
    valid_channel=1,  # Which channel to use, recent IR data always uses 1
    # Recent visible data uses 0
)

mpf.recalculate_wavelengths(8.8)
# %%

para, perp, iso = mpf.avg_and_concat()

pol_ds = dataset.PolTRSpec(para, perp)
pol_ds.subtract_background()
merged_ds = pol_ds.merge_nearby_channels(8)
merged_ds.plot.spec(-.5, n_average=3);

# %%
merged_ds.plot.spec(-1.5, n_average=1);



# %%
merged_ds.t

# %%
